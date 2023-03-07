from operator import itemgetter
from typing import Dict, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target

class BPPOPretrainPolicy(BasePolicy):
    def __init__(
        self, 
        actor: BasePolicy, 
        critic_v: Critic, 
        critic_q: Critic, 
        actor_optim: torch.optim.Optimizer, 
        critic_v_optim: torch.optim.Optimizer, 
        critic_q_optim: torch.optim.Optimizer, 
        discount: float=0.99, 
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic_v = critic_v
        self.critic_q = critic_q
        self.actor_optim = actor_optim
        self.critic_v_optim = critic_v_optim
        self.critic_q_optim = critic_q_optim
        
        self.critic_q_target = make_target(self.critic_q)
        
        self._discount = discount
        
    def pretrain_update_v(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obss, returns = itemgetter("observations", "returns")(batch)
        obss, returns = convert_to_tensor(obss), convert_to_tensor(returns)
        loss = torch.nn.functional.mse_loss(self.critic_v(obss), returns)
        self.critic_v_optim.zero_grad()
        loss.backward()
        self.critic_v_optim.step()
        return {"pretrain/v_loss": loss.item()}
    
    def pretrain_update_q(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obss, actions, next_obss, next_actions, rewards, terminals = \
            itemgetter("observations", "actions", "next_observations", "next_actions", "rewards", "terminals")(batch)
        with torch.no_grad():
            q_target = rewards + self._discount * (1-terminals) * self.critic_q_target(next_obss, next_actions)
        loss = torch.nn.functional.mse_loss(self.critic_q(obss, actions), q_target)
        self.critic_q_optim.zero_grad()
        loss.backward()
        self.critic_q_optim.step()
        return {"pretrain/q_loss", loss.item()}
    
    def pretrain_update_actor(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obss, actions = itemgetter("observations", "actions")(batch)
        loss = (-self.actor.evaluate(obss, actions)[0]).mean()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        return {"pretrain/actor_loss", loss.item()}
    

class BPPOPolicy(BasePolicy):
    def __init__(
        self, 
        actor: BaseActor, 
        critic_v: Critic, 
        critic_q: Critic,
        actor_optim: torch.optim.Optimizer, 
        clip_ratio: float, 
        entropy_weight: float,  
        omega: float
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic_v = critic_v
        self.critic_q = critic_q
        self.actor_optim = actor_optim
        
        self.old_actor = make_target(actor)
        
        self._clip_ratio = clip_ratio
        self._entropy_weight = entropy_weight
        self._omega = omega
        self._count = 0
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obss = itemgetter("observations")(batch)
        self._count += 1
        clip_ratio = self._clip_ratio(self._count)

        old_actions, old_logprobs, *_ = self.old_actor.sample(obss)
        adv = self.critic_q(obss, old_actions) - self.critic_v(obss)
        adv = (adv-adv.mean()) / (adv.std()+1e-10)
        adv = adv * torch.where(adv > 0, self._omega, 1-self._omega)
        
        new_logprobs, info = self.actor.evaluate(obss, old_actions, return_dist=True)
        ratio = (new_logprobs-old_logprobs).exp()
        
        loss1 = ratio * adv
        loss2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)*adv
        actor_loss = -torch.min(loss1, loss2).mean()
        
        entropy_loss = - info["dist"].entropy().sum(-1, keepdim=True).mean()
        
        self.actor_optim.zero_grad()
        (actor_loss + self._entropy_weight * entropy_loss).backward()
        self.actor_optim.step()
        return {"bppo_loss/actor_loss": actor_loss.item(), "bppo_loss/entropy_loss": entropy_loss.item()}
        
        
        