from copy import deepcopy
from operator import itemgetter
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import DeterministicActor, GaussianActor
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.functional import expectile_regression
from offlinerllib.utils.misc import convert_to_tensor, make_target


class IQLPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_q: nn.Module, 
        critic_v: nn.Module,
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

        self._max_action = max_action
        self._tau = tau
        self._discount = discount
        self._expectile = expectile
        self._temperature = temperature
        
        self.to(device)
    
    def configure_optimizers(self, actor_lr, critic_v_lr, critic_q_lr, actor_opt_scheduler_steps=None):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=critic_v_lr)
        self.critic_q_optim = torch.optim.Adam(self.critic_q.parameters(), lr=critic_q_lr)
        if actor_opt_scheduler_steps is not None:
            self.actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim, actor_opt_scheduler_steps)
        else:
            self.actor_lr_scheduler = None
    
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, *_ = self.actor.sample(obs, deterministic)
        return action.squeeze().cpu().numpy()
    
    def update(self, batch: Dict) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        obss, actions, next_obss, rewards, terminals = itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        
        # do the inference
        with torch.no_grad():
            q_old = self.critic_q_old(obss, actions)
            v_old = self.critic_v(obss)
            next_v_old = self.critic_v(next_obss)

        # update v
        v = self.critic_v(obss)
        critic_v_loss = expectile_regression(v, q_old, self._expectile).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()
        
        # update q
        target_q = rewards + self._discount * (1 - terminals) * next_v_old
        q_both = self.critic_q(obss, actions, reduce=None)
        critic_q_loss = (q_both - target_q).pow(2).sum(0).mean()

        self.critic_q_optim.zero_grad()
        critic_q_loss.backward()
        self.critic_q_optim.step()

        # update actor
        advantage = q_old - v_old
        exp_advanrage = (self._temperature * advantage).exp().clamp(max=100.0)
        if isinstance(self.actor, DeterministicActor):
            # use bc loss
            policy_out = torch.sum((self.actor.sample(obss)[0] - actions)**2, dim=1)
        elif isinstance(self.actor, GaussianActor):
            policy_out = - self.actor.evaluate(obss, actions)[0]
        actor_loss = (exp_advanrage * policy_out).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q": critic_q_loss.item(),
            "loss/v": critic_v_loss.item()
        }
        
    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q_old.parameters(), self.critic_q.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            