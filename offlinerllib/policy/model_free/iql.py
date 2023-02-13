import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import make_target
from offlinerllib.module.actor import DeterminisitcActor, SquashedGaussianActor


class IQLPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        max_action: float = 1.0, 
        expectile: float = 0.8,
        temperature: float = 0.1, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q1 = critic_q1
        self.critic_q1_old = make_target(self.critic_q1)
        self.critic_q2 = critic_q2
        self.critic_q2_old = make_target(self.critic_q2)
        self.critic_v = critic_v

        self.actor_optim = actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim
        self.critic_v_optim = critic_v_optim

        self._max_action = max_action
        self._tau = tau
        self._gamma = gamma
        self._expectile = expectile
        self._temperature = temperature
        
        self.to(device)

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q1_old.parameters(), self.critic_q1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_q2_old.parameters(), self.critic_q2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.from_numpy(obs).to(self.device)
        action, *_ = self.actor.sample(obs, deterministic)
        return np.clip(action.cpu().numpy(), -self._max_action, self._max_action)
    
    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        return weight * (diff**2)
    
    def update(self, batch: Dict) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = torch.from_numpy(_value).to(self.device)
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update critic
        q1, q2 = self.critic_q1(obss, actions), self.critic_q2(obss, actions)
        with torch.no_grad():
            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma * (1 - terminals) * next_v
        
        critic_q1_loss = ((q1 - target_q).pow(2)).mean()
        critic_q2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()
        
        # update value net
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
        v = self.critic_v(obss)
        critic_v_loss = self._expectile_regression(q-v).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        # update actor
        with torch.no_grad():
            advantage = q - v
            exp_advanrage = (self._temperature * advantage).exp().clamp(max=100.0)
        if isinstance(self.actor, DeterminisitcActor):
            # use bc loss
            policy_out = torch.sum((self.actor.sample(obss) - actions)**2, dim=1)
        elif isinstance(self.actor, SquashedGaussianActor):
            policy_out = - self.actor.evaluate(obss, actions)
        actor_loss = (exp_advanrage * policy_out).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q1": critic_q1_loss.item(),
            "loss/q2": critic_q2_loss.item(),
            "loss/v": critic_v_loss.item()
        }