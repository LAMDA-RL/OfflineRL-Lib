from operator import itemgetter
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target
from offlinerllib.utils.functional import gumbel_rescale_loss


class XSACPolicy(BasePolicy):
    """
    Soft Actor-Critic with Gumbel Regression (Extreme Soft Actor-Critic) <Ref>: Extreme Q-Learning <Ref: https://arxiv.org/abs/2301.02328>
    """

    def __init__(
        self, 
        actor: BaseActor, 
        critic_q: Critic, 
        critic_v: Critic, 
        loss_temperature: float, 
        actor_update_freq: int, 
        critic_q_update_freq: int, 
        critic_v_update_freq: int,
        target_update_freq: int, 
        alpha: Union[float, Tuple[float, float]]=0.2, 
        tau: float=0.005, 
        discount: float=0.99, 
        clip_max: float=10.0, 
        device: Union[str, torch.device]="cpu"
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic_q = critic_q
        self.critic_v = critic_v
        self.critic_v_target = make_target(critic_v)
        
        self._actor_update_freq = actor_update_freq
        self._critic_q_update_freq = critic_q_update_freq
        self._critic_v_update_freq = critic_v_update_freq
        self._target_update_freq = target_update_freq

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            target_entropy, alpha_lr = alpha
            self._log_alpha = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device), requires_grad=True)
            self._target_entropy = target_entropy
            self.alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = torch.tensor([alpha], dtype=torch.float32, device=device, requires_grad=False)

        
        self._loss_temperature = loss_temperature
        self._tau = tau
        self._discount = discount
        self._clip_max = clip_max
        self._steps = 0
        self.to(device) 

    def configure_optimizers(self, learning_rate):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_q_optim = torch.optim.Adam(self.critic_q.parameters(), lr=learning_rate)
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=learning_rate)
        
    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, _, _ = self.actor.sample(obs, deterministic)
        return action.squeeze(0).cpu().numpy()
    
    def _critic_q_loss(self, obss, actions, next_obss, rewards, terminals) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            q_target = self.critic_v_target(next_obss)
            q_target = rewards + self._discount * (1-terminals) * q_target
        q_pred = self.critic_q(obss, actions)
        q_loss = (q_pred - q_target).pow(2).sum(0).mean()
        return q_loss, {"loss/q_loss": q_loss.item(), "misc/q_pred": q_pred.mean().item()}
    
    def _critic_v_loss(self, obss, new_q_pred) -> Tuple[torch.Tensor, Dict[str, float]]:
        new_q_pred = new_q_pred.detach()
        v_pred = self.critic_v(obss)
        v_loss = gumbel_rescale_loss(v_pred, new_q_pred, alpha=self._loss_temperature, clip_max=self._clip_max).mean()
        return v_loss, {"loss/v_loss": v_loss.item(), "misc/v_pred": v_pred.mean().item()}

    def _actor_loss(self, new_logprob, new_q_pred) -> Tuple[torch.Tensor, Dict[str, float]]:
        actor_loss = (self._alpha * new_logprob - new_q_pred).mean()
        return actor_loss, {"loss/actor_loss": actor_loss.item()}
    
    def _alpha_loss(self, new_logprob) -> torch.Tensor:
        alpha_loss = -(self._log_alpha * (new_logprob + self._target_entropy).detach()).mean()
        return alpha_loss
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obss, actions, next_obss, rewards, terminals = \
            [convert_to_tensor(t, self.device) for t in itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)]
        metrics = {}
    
        # update Q
        if self._steps % self._critic_q_update_freq == 0:
            q_loss, q_metrics = self._critic_q_loss(obss, actions, next_obss, rewards, terminals)
            metrics.update(q_metrics)
            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()
        
        # get Q value for subsequent updates
        new_action, new_logprob, *_= self.actor.sample(obss)
        new_q_pred = self.critic_q(obss, new_action).min(0)[0]
        
        # update V
        if self._steps % self._critic_v_update_freq == 0:
            v_loss, v_metrics = self._critic_v_loss(obss, new_q_pred)
            metrics.update(v_metrics)
            self.critic_v_optim.zero_grad()
            v_loss.backward()
            self.critic_v_optim.step()
            
        # update actor
        if self._steps % self._actor_update_freq == 0:
            actor_loss, actor_metrics = self._actor_loss(new_logprob, new_q_pred)
            metrics.update(actor_metrics)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            if self._is_auto_alpha:
                alpha_loss = self._alpha_loss(new_logprob)
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self._alpha = self._log_alpha.exp().detach()
                alpha_loss = alpha_loss.item()
            else:
                alpha_loss = 0
            metrics.update({
                "misc/alpha": self._alpha.item(), 
                "loss/alpha_loss": alpha_loss
            })
            
        # update target
        if self._steps % self._target_update_freq == 0:
            for o, n in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
                o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        
        self._steps += 1
        return metrics
    
