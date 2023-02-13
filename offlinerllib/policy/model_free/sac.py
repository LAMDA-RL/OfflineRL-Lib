import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from typing import Dict, Union, Tuple, Any

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import make_target

class SACPolicy(BasePolicy):
    """
    Soft Actor Critic <Ref: https://arxiv.org/abs/1801.01290>
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
        gamma: float = 0.99, 
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.critic1_old = make_target(self.critic1)
        self.critic2_old = make_target(self.critic2)

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
            
        self.to(device)
        
    @torch.no_grad()   
    def select_action(self, obs: np.ndarray, deterministic: bool=False):
        obs = torch.from_numpy(obs).to(self.device)
        action, _, _ = self.actor.sample(obs, deterministic)
        return action.squeeze().cpu().numpy()
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for _key, _value in batch.items():
            batch[_key] = torch.from_numpy(batch[_value]).to(self.device)
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actor.sample(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["misc/alpha"] = self._alpha.item()

        return result
    

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

