from operator import itemgetter
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target


class SACVPolicy(BasePolicy):
    """
    Soft Actor Critic with Value network <Ref: https://arxiv.org/abs/1801.01290>
    """

    def __init__(
        self,
        actor: BaseActor,
        critic_q: Critic,
        critic_v: Critic,
        tau: float = 0.005,
        discount: float = 0.99,
        alpha: Union[float, Tuple[float, float]] = 0.2,
        target_update_freq: int=1, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q = critic_q
        self.critic_v = critic_v
        self.critic_q_target = make_target(self.critic_q)
        self.critic_v_target = make_target(self.critic_v)

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            target_entropy, alpha_lr = alpha
            self._log_alpha = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device), requires_grad=True)
            self._target_entropy = target_entropy
            self.alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = torch.tensor([alpha], dtype=torch.float32, device=device, requires_grad=False)

        self._tau = tau
        self._discount = discount
        self._target_update_freq = target_update_freq
        self._steps = 0

        self.to(device)

    def configure_optimizers(self, actor_lr, critic_q_lr, critic_v_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_q_optim = torch.optim.Adam(self.critic_q.parameters(), lr=critic_q_lr)
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=critic_v_lr)

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, _, _ = self.actor.sample(obs, deterministic)
        return action.squeeze(0).cpu().numpy()

    def _actor_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obss = batch["observations"]
        new_actions, new_logprobs, _ = self.actor.sample(obss)
        q_value = self.critic_q(obss, new_actions)
        v_value = self.critic_v(obss)
        actor_loss = (self._alpha * new_logprobs - (q_value - v_value)).mean()
        return actor_loss, {}

    def _critic_q_loss(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obss, actions, next_obss, rewards, terminals = \
            itemgetter("observations", "actions", "next_observations", "rewards","terminals")(batch)
        with torch.no_grad():
            next_v_value = self.critic_v_target(next_obss)
            target_q = rewards + self._discount * (1 - terminals) * next_v_value
        q_value = self.critic_q(obss, actions)
        critic_q_loss = (q_value - target_q).pow(2).mean()
        return critic_q_loss, {}

    def _critic_v_loss(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obss = batch["observations"]
        v_value = self.critic_v(obss)
        with torch.no_grad():
            new_actions, new_logprobs, _ = self.actor.sample(obss)
            q_value = self.critic_q(obss, new_actions)
            target_v = q_value - self._alpha * new_logprobs
        critic_v_loss = (v_value - target_v).pow(2).mean()
        return critic_v_loss, {}

    def _alpha_loss(self, obss: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, new_logprobs, _ = self.actor.sample(obss)
        alpha_loss = -(self._log_alpha * (new_logprobs + self._target_entropy)).mean()
        return alpha_loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
            
        metrics = {}
        obss = batch["observations"]

        # critic q update
        critic_q_loss, critic_q_loss_metrics = self._critic_q_loss(batch)
        metrics.update(critic_q_loss_metrics)
        self.critic_q_optim.zero_grad()
        critic_q_loss.backward()
        self.critic_q_optim.step()

        # critic v update
        critic_v_loss, critic_v_loss_metrics = self._critic_v_loss(batch)
        metrics.update(critic_v_loss_metrics)
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        # actor update
        actor_loss, actor_loss_metrics = self._actor_loss(batch)
        metrics.update(actor_loss_metrics)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha update
        if self._is_auto_alpha:
            alpha_loss = self._alpha_loss(obss)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.exp().detach()
        else:
            alpha_loss = 0
        metrics["misc/alpha"] = self._alpha.item()

        if self._steps % self._target_update_freq == 0:
            self._sync_weight()

        # update info
        self._steps += 1
        metrics.update({
            "loss/actor": actor_loss.item(),
            "loss/critic_q": critic_q_loss.item(),
            "loss/critic_v": critic_v_loss.item(),
            "loss/alpha": alpha_loss
        })
        return metrics

    def _sync_weight(self) -> None:
        for target_param, param in zip(self.critic_q_target.parameters(), self.critic_q.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) + param.data * self._tau)
        for target_param, param in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) + param.data * self._tau)
