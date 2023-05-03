from operator import itemgetter
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import DeterministicActor, GaussianActor, CategoricalActor
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.functional import gumbel_log_loss, gumbel_rescale_loss
from offlinerllib.utils.misc import convert_to_tensor, make_target


class XQLPolicy(BasePolicy):
    """
    Extreme Q-Learning <Ref: https://arxiv.org/abs/2301.02328>
    """
    
    def __init__(
        self, 
        actor: nn.Module, 
        critic_q: nn.Module, 
        critic_v: nn.Module, 
        num_v_update: int=1, 
        scale_random_sample: int=0, 
        loss_temperature: float=1.0, 
        aw_temperature: float=0.1, 
        use_log_loss: bool=False, 
        noise_std: float=0,
        tau: float = 0.005, 
        discount: float = 0.99, 
        max_action: float = 1.0, 
        max_clip: float = 1.0, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic_q = critic_q
        self.critic_q_target = make_target(self.critic_q)
        self.critic_v = critic_v
        
        self.num_v_update = num_v_update
        self.scale_random_sample = scale_random_sample
        
        self.tau = tau
        self.discount = discount
        self.loss_temperature = loss_temperature
        self.aw_temperature = aw_temperature
        self.use_log_loss = use_log_loss
        self.max_action = max_action
        self.max_clip = max_clip
        self.noise_std = noise_std

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
    def select_action(self, obs: np.ndarray, deterministic: bool=False):
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, *_ = self.actor.sample(obs, deterministic)
        return action.squeeze().cpu().numpy()
        
    def update(self, batch: Dict) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        obss, actions, next_obss, rewards, terminals = itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        
        # update value network for num_v_update times
        v_loss_value = 0
        for _ in range(self.num_v_update):
            if self.scale_random_sample > 0:
                random_actions = torch.rand([self.scale_random_sample*actions.shape[0], *actions.shape[1:]]).to(self.device) * self.max_action * 2 - self.max_action
                v_obss = torch.concat([obss]*(self.scale_random_sample+1), dim=0)
                v_actions = torch.cat([actions, random_actions], dim=0)
            else:
                v_obss = obss
                v_actions = actions
            if self.noise_std > 0:
                noise = (torch.randn_like(v_actions)*self.noise_std).clamp(-0.5*self.max_action, 0.5*self.max_action)
                v_actions = (v_actions + noise).clamp(-self.max_action, self.max_action)
                
            with torch.no_grad():
                q = self.critic_q_target(v_obss, v_actions)
            v = self.critic_v(v_obss)
            if self.use_log_loss:
                value_loss = gumbel_log_loss(v, q, alpha=self.loss_temperature, clip_max=self.max_clip).mean()
            else:
                value_loss = gumbel_rescale_loss(v, q, alpha=self.loss_temperature, clip_max=self.max_clip).mean()
            clip_ratio = (((q-v)/self.loss_temperature) > self.max_clip).float().mean().item()
            self.critic_v_optim.zero_grad()
            value_loss.backward()
            self.critic_v_optim.step()
            v_loss_value += value_loss.detach().cpu().item()
        v_loss_value /= self.num_v_update
            
        # update actor network
        with torch.no_grad():
            v = self.critic_v(obss)
            q = self.critic_q_target(obss, actions)
            exp_advantage = torch.exp((q-v)*self.aw_temperature).clamp(max=100.0)
        if isinstance(self.actor, DeterministicActor):
            policy_out = torch.sum((self.actor.sample(obss)[0] - actions)**2, dim=1)
        elif isinstance(self.actor, (GaussianActor, CategoricalActor)):
            policy_out = - self.actor.evaluate(obss, actions)[0]
        actor_loss = (exp_advantage * policy_out).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()
        actor_loss_value = actor_loss.detach().cpu().item()
        
        # update q network
        with torch.no_grad():
            next_v = self.critic_v(next_obss)
            target_q = rewards + self.discount * (1-terminals) * next_v
        q = self.critic_q(obss, actions, reduce=False)
        q_loss = 2*torch.nn.functional.huber_loss(q.reshape([-1, 1]), torch.tile(target_q, [2, 1]), delta=20.0)
        self.critic_q_optim.zero_grad()
        q_loss.backward()
        self.critic_q_optim.step()
        q_loss_value = q_loss.detach().cpu().item()
        
        self._sync_weight()
        
        return {
            "loss/q_loss": q_loss_value, 
            "loss/actor_loss": actor_loss_value, 
            "loss/v_loss": v_loss_value, 
            "misc/clip_ratio": clip_ratio
        }
                
    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q_target.parameters(), self.critic_q.parameters()):
            o.data.copy_(o.data * (1.0 - self.tau) + n.data * self.tau)

