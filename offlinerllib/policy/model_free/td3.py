from operator import itemgetter
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target


class TD3Policy(BasePolicy):
    """
    Twin Delayed Deep Deterministic Policy Gradient <Ref: https://arxiv.org/abs/1802.09477>
    """

    def __init__(
        self,
        actor: BaseActor,
        critic: Critic, 
        actor_update_interval: int = 2, 
        policy_noise: float = 0.2, 
        noise_clip: float = 0.5,
        exploration_noise: Any = None, 
        tau: float = 0.005,
        discount: float = 0.99,
        max_action: float = 1.0, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.actor_target = make_target(self.actor)
        self.critic = critic
        self.critic_target = make_target(self.critic)

        self._tau = tau
        self._discount = discount
        self._actor_update_interval = actor_update_interval
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._exploration_noise = exploration_noise
        self._max_action = max_action

        self._update_cnt = 0
        
        self.to(device)

    def configure_optimizers(self, actor_lr, critic_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    @torch.no_grad()
    def select_action(
        self, 
        obs: np.ndarray, 
        deterministic: bool=False
    ) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action = self.actor.sample(obs, deterministic)[0].squeeze(0).cpu().numpy()
        if not deterministic and self._exploration_noise is not None:
            action = np.clip(action + self._exploration_noise(action.shape), -self._max_action, self._max_action)
        return action
    
    def critic_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obss, actions, next_obss, rewards, terminals = \
            itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        q_both = self.critic(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_target.sample(next_obss)[0] + noise).clamp(-self._max_action, self._max_action)
            q_next = self.critic_target(next_obss, next_actions).min(0)[0]
            q_target = rewards + self._discount * (1-terminals) * q_next
        critic_loss = (q_both - q_target).pow(2).sum(0).mean()
        return critic_loss, {"loss/q_loss": critic_loss.item()}
    
    def actor_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obss = itemgetter("observations")(batch)
        new_actions, *_ = self.actor.sample(obss)
        new_q1 = self.critic(obss, new_actions)[0, ...]
        actor_loss = -new_q1.mean()
        return actor_loss, {"loss/actor_loss": actor_loss.item()}
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        metrics = {}
        self._update_cnt += 1

        critic_loss, critic_loss_metrics = self.critic_loss(batch)
        metrics.update(critic_loss_metrics)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        if self._update_cnt % self._actor_update_interval == 0:
            actor_loss, actor_loss_metrics = self.actor_loss(batch)
            metrics.update(actor_loss_metrics)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            self._sync_weight()
        
        return metrics

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_target.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_target.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
