from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from offlinerlkit.utils.noise import GaussianNoise

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import make_target


class TD3Policy(BasePolicy):
    """
    Twin Delayed Deep Deterministic policy gradient <Ref: https://arxiv.org/abs/1802.09477>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.actor_old = make_target(self.actor)
        self.actor_optim = actor_optim

        self.critic1 = critic1
        self.critic1_old = make_target(self.critic1)
        self.critic1_optim = critic1_optim

        self.critic2 = critic2
        self.critic2_old = make_target(self.critic2)
        self.critic2_optim = critic2_optim

        self._tau = tau
        self._gamma = gamma

        self._max_action = max_action
        self.exploration_noise = exploration_noise
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._freq = update_actor_freq

        self._cnt = 0
        self._last_actor_loss = 0
        
        self.to(device)

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.from_numpy(obs).to(self.device)
        action = self.actor.sample(obs)
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = torch.from_numpy(_value).to(self.device)
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic1(obss, a)
            actor_loss = -q.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._sync_weight()
        
        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }