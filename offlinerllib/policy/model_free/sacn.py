import torch
import torch.nn as nn
import numpy as np

from operator import itemgetter

from copy import deepcopy
from typing import Dict, Union, Tuple, Any

from offlinerllib.policy import BasePolicy
from offlinerllib.policy.model_free.sac import SACPolicy
from offlinerllib.utils.misc import make_target
from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic


class SACNPolicy(SACPolicy):
    def __init__(
        self, 
        actor: BaseActor, 
        critic: Critic, 
        actor_optim: nn.Module, 
        critic_optim: nn.Module, 
        tau: float = 0.005, 
        gamma: float = 0.99, 
        alpha: Union[float, Tuple[float, float]] = 0.2, 
        do_reverse_update: bool = False, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__(actor, critic, actor_optim, critic_optim, tau, gamma, alpha, device)
        self.do_reverse_update = do_reverse_update
    
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return super().select_action(obs, deterministic)
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.do_reverse_update:
            return self.reverse_update(batch)
        else:
            return super().update(batch)
        
    def reverse_update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for _key, _value in batch.items():
            batch[_key] = torch.from_numpy(_value).to(self.device)
        obss, actions, next_obss, rewards, terminals = itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        
        # update entropy
        if self._is_auto_alpha:
            with torch.no_grad():
                new_actions, new_logprobs, _ = self.actor.sample(obss)
            alpha_loss = -(self._log_alpha * (self._target_entropy + new_logprobs)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # update actor
        new_actions, new_logprobs, _ = self.actor.sample(obss)
        q_values = self.critic(obss, new_actions)
        q_values_min = torch.min(q_values, dim=0)[0]
        q_values_std = torch.std(q_values, dim=0).mean().item()
        actor_loss = (self._alpha * new_logprobs - q_values_min).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # critic update
        with torch.no_grad():
            next_actions, next_logprobs, _ = self.actor.sample(next_obss)
            target_q = self.critic_target(next_obss, next_actions).min(0)[0] - self._alpha * next_logprobs
            target_q = rewards + self._gamma * (1-terminals) * target_q
        q_values = self.critic(obss, actions)
        critic_loss = (q_values - target_q).pow(2).sum(0).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        self._sync_weight()
        
        result = {
            "loss/actor": actor_loss.item(), 
            "loss/critic": critic_loss.item(), 
            "misc/alpha": self._alpha.item(), 
            "misc/q_values_std": q_values_std, 
            "misc/q_values_min": q_values_min.item(), 
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
        return result
            
        
        
            