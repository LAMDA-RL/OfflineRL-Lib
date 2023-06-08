from operator import itemgetter
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target


class SACPolicy(BasePolicy):
    """
    Soft Actor Critic <Ref: https://arxiv.org/abs/1801.01290>
    """

    def __init__(
        self,
        actor: BaseActor,
        critic: Critic,
        tau: float = 0.005,
        discount: float = 0.99,
        alpha: Union[float, Tuple[float, float]] = 0.2,
        target_update_freq: int=1, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.critic_target = make_target(self.critic)

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

    def configure_optimizers(self, actor_lr, critic_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

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
        obss, actions, next_obss, rewards, terminals = \
            itemgetter("observations", "actions", "next_observations", "rewards","terminals")(batch)
        new_actions, new_logprobs, _ = self.actor.sample(obss)
        q_values = self.critic(obss, new_actions)
        if len(q_values.shape) == 2:
            q_values = q_values.unsqueeze(0)
        q_values_min = torch.min(q_values, dim=0)[0]
        q_values_std = torch.std(q_values, dim=0).mean().item()
        q_values_mean = q_values.mean().item()
        actor_loss = (self._alpha * new_logprobs - q_values_min).mean()
        return actor_loss,  {"misc/q_values_std": q_values_std, "misc/q_values_min": q_values_min.mean().item(), "misc/q_values_mean": q_values_mean}

    def _critic_loss(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obss, actions, next_obss, rewards, terminals = \
            itemgetter("observations", "actions", "next_observations", "rewards","terminals")(batch)
        with torch.no_grad():
            next_actions, next_logprobs, _ = self.actor.sample(next_obss)
            target_q = self.critic_target(next_obss, next_actions).min(0)[0] - self._alpha * next_logprobs
            target_q = rewards + self._discount * (1 - terminals) * target_q
        q_values = self.critic(obss, actions)
        
        critic_loss = (q_values - target_q).pow(2).sum(0).mean()
        return critic_loss, {}

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

        # critic update
        critic_loss, critic_loss_metrics = self._critic_loss(batch)
        metrics.update(critic_loss_metrics)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

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
            "loss/critic": critic_loss.item(),
            "loss/alpha": alpha_loss
        })
        return metrics

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_target.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
