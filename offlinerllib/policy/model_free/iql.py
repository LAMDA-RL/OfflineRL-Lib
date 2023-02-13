import numpy as np
import torch
import torch.nn as nn
import gym
from operator import itemgetter

from copy import deepcopy
from typing import Dict, Union, Tuple

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import make_target
from offlinerllib.module.actor import DeterministicActor, GaussianActor
from offlinerllib.utils.functional import expectile_regression


class IQLPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_q: nn.Module, 
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q_optim: torch.optim.Optimizer, 
        critic_v_optim: torch.optim.Optimizer,
        expectile: float = 0.7,
        temperature: float = 3.0, 
        tau: float = 0.005,
        discount: float = 0.99,
        max_action: float = 1.0, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q = critic_q
        self.critic_q_old = make_target(self.critic_q)
        self.critic_v = critic_v

        self.actor_optim = actor_optim
        self.critic_q_optim = critic_q_optim
        self.critic_v_optim = critic_v_optim

        self._max_action = max_action
        self._tau = tau
        self._discount = discount
        self._expectile = expectile
        self._temperature = temperature
        
        self.to(device)
    
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, *_ = self.actor.sample(obs, deterministic)
        return action.squeeze().cpu().numpy()
    
    def update(self, batch: Dict) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = torch.from_numpy(_value).to(self.device)
        obss, actions, next_obss, rewards, terminals = itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        
        # update v
        with torch.no_grad():
            q = self.critic_q_old(obss, actions)
        v = self.critic_v(obss)
        critic_v_loss = expectile_regression(v, q, self._expectile).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()
        
        # update q
        with torch.no_grad():
            next_v = self.critic_v(next_obss)
            target_q = rewards + self._discount * (1 - terminals) * next_v
        q_both = self.critic_q(obss, actions, reduce=None)
        critic_q_loss = (q_both - target_q).pow(2).sum(0).mean()

        self.critic_q_optim.zero_grad()
        critic_q_loss.backward()
        self.critic_q_optim.step()

        # update actor
        with torch.no_grad():
            v = self.critic_v(obss)    # theoretically we can re-use v from above, however the original implementation re-computes v with the updated critic_v, so we keep the same
            advantage = q - v
            exp_advanrage = (self._temperature * advantage).exp().clamp(max=100.0)
        if isinstance(self.actor, DeterministicActor):
            # use bc loss
            policy_out = torch.sum((self.actor.sample(obss) - actions)**2, dim=1)
        elif isinstance(self.actor, GaussianActor):
            policy_out = - self.actor.evaluate(obss, actions)[0]
        actor_loss = (exp_advanrage * policy_out).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q": critic_q_loss.item(),
            "loss/v": critic_v_loss.item()
        }
        
    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q_old.parameters(), self.critic_q.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            