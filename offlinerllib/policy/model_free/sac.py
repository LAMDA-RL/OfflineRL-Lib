import torch
import torch.nn as nn
import numpy as np

from operator import itemgetter

from copy import deepcopy
from typing import Dict, Union, Tuple, Any

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import make_target
from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic

class SACPolicy(BasePolicy):
    """
    Soft Actor Critic <Ref: https://arxiv.org/abs/1801.01290>
    """
    def __init__(
        self, 
        actor: BaseActor, 
        critic: Critic, 
        actor_optim: torch.optim.Optimizer, 
        critic_optim: torch.optim.Optimizer, 
        tau: float = 0.005, 
        gamma: float = 0.99, 
        alpha: Union[float, Tuple[float, float]] = 0.2, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.critic_target = make_target(self.critic)

        self.actor_optim = actor_optim
        self.critic_optim = critic_optim

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            target_entropy, alpha_lr = alpha
            self._log_alpha = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=self.device), requires_grad=True)
            self._target_entropy = target_entropy
            self.alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = torch.tensor([alpha], dtype=torch.float32, device=device, requires_grad=False)
        
        self._tau = tau
        self._gamma = gamma
            
        self.to(device)
        
    @torch.no_grad()   
    def select_action(self, obs: np.ndarray, deterministic: bool=False) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, _, _ = self.actor.sample(obs, deterministic)
        return action.squeeze().cpu().numpy()
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for _key, _value in batch.items():
            batch[_key] = torch.from_numpy(_value).to(self.device)
        obss, actions, next_obss, rewards, terminals = itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        
        # update critic
        q_both = self.critic(obss, actions)
        with torch.no_grad():
            next_actions, next_logprobs, _ = self.actor.sample(next_obss)
            target_q = self.critic_target(next_obss, next_actions).min(0)[0] - self._alpha * next_logprobs
            target_q = rewards + self._gamma * (1 - terminals) * target_q

        critic_loss = (q_both - target_q).pow(2).sum(0).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        new_actions, new_logprobs, _ = self.actor.sample(obss)
        actor_loss = (self._alpha*new_logprobs - self.critic(obss, new_actions).min(0)[0]).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            alpha_loss = -(self._log_alpha * (new_logprobs + self._target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(), 
            "misc/alpha": self._alpha.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()

        return result
    

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_target.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

