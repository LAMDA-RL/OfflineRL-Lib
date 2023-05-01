from operator import itemgetter
from typing import Dict, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target

class InACPolicy(BasePolicy):
    """
    In-Sample Actor Critic <Ref: https://arxiv.org/abs/2302.14372>
    """
    def __init__(
        self, 
        actor: BaseActor, 
        behavior: BaseActor, 
        critic_q: Critic,
        critic_v: Critic,
        temperature: float = 0.01, 
        discount: float = 0.99, 
        tau: float = 5e-3, 
        adv_min: float = 1e-8, 
        adv_max: float = 1e4, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.actor = actor
        self.behavior = behavior
        self.critic_q = critic_q
        self.critic_v = critic_v
        self.critic_q_target = make_target(critic_q)
        self._temperature = temperature
        self._tau = tau
        self._discount = discount
        self._adv_min = adv_min
        self._adv_max = adv_max
        
        self.to(device)

    def configure_optimizers(self, actor_lr, critic_q_lr, critic_v_lr, behavior_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_q_optim = torch.optim.Adam(self.critic_q.parameters(), lr=critic_q_lr)
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=critic_v_lr)
        self.behavior_optim = torch.optim.Adam(self.behavior.parameters(), lr=behavior_lr)
        
    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, _, _ = self.actor.sample(obs, deterministic)
        return action.squeeze().cpu().numpy()
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for _key in batch:
            batch[_key] = convert_to_tensor(batch[_key], device=self.device)
        metrics = {}
            
        # update behavior policy
        behavior_loss, behavior_metrics = self.behavior_loss(batch)
        self.behavior_optim.zero_grad()
        behavior_loss.backward()
        self.behavior_optim.step()
        metrics.update(behavior_metrics)
        
        # update value network
        v_loss, v_metrics = self.v_loss(batch)
        self.critic_v_optim.zero_grad()
        v_loss.backward()
        self.critic_v_optim.step()
        metrics.update(v_metrics)
        
        # udpate q network
        q_loss, q_metrics = self.q_loss(batch)
        self.critic_q_optim.zero_grad()
        q_loss.backward()
        self.critic_q_optim.step()
        metrics.update(q_metrics)

        # update actor network
        actor_loss, actor_metrics = self.actor_loss(batch)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        metrics.update(actor_metrics)

        self._sync_weight()

        return metrics
        
    def behavior_loss(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obss, actions = itemgetter("observations", "actions")(batch)
        behavior_loss = - self.behavior.evaluate(obss, actions)[0].mean()
        return behavior_loss, {"loss/behavior_loss": behavior_loss.item()}
    
    def v_loss(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # V only functions as a baseline
        obss = itemgetter("observations")(batch)
        with torch.no_grad():
            new_actions, new_logprobs, *_ = self.actor.sample(obss)
            v_target = self.critic_q_target(obss, new_actions).min(0)[0] - self._temperature * new_logprobs
        v = self.critic_v(obss)
        v_loss = torch.nn.functional.mse_loss(v, v_target)
        return v_loss, {"loss/v_loss": v_loss.item()}
    
    def q_loss(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obss, actions, rewards, next_obss, terminals = \
            itemgetter("observations", "actions", "rewards", "next_observations", "terminals")(batch)
        with torch.no_grad():
            next_actions, next_logprobs, *_ = self.actor.sample(next_obss)
            q_target = self.critic_q_target(next_obss, next_actions).min(0)[0] - self._temperature * next_logprobs
            q_target = rewards + self._discount * (1-terminals) * q_target
        q_both = self.critic_q(obss, actions)
        q_loss = (q_both - q_target).pow(2).sum(0).mean()
        return q_loss, {"loss/q_loss": q_loss.item()}
    
    def actor_loss(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obss, actions = itemgetter("observations", "actions")(batch)
        logprobs, *_ = self.actor.evaluate(obss, actions)
        with torch.no_grad():
            min_q = self.critic_q(obss, actions).min(0)[0]
            value = self.critic_v(obss)
            behavior_logprob, *_ = self.behavior.evaluate(obss, actions)
            clipped = torch.clip(torch.exp((min_q - value) / self._temperature - behavior_logprob), self._adv_min, self._adv_max)
            
        actor_loss = - (clipped * logprobs).mean()
        return actor_loss, {"loss/actor_loss": actor_loss.item()}

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q_target.parameters(), self.critic_q.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
            
            