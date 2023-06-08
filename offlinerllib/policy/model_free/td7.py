from operator import itemgetter
from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import copy

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target


class TD7Policy(BasePolicy):
    """
    For Sale: State-Action Representation Learning for Deep Reinforcement Learning <Ref: https://arxiv.org/pdf/2306.02451v1.pdf>
    """
    
    def __init__(
        self, 
        actor: BaseActor, 
        critic: Critic, 
        encoder: nn.Module, 
        offline: bool=False, 
        actor_update_interval: int=2, 
        target_update_interval: int=250, 
        policy_noise: float=0.2, 
        noise_clip: float=0.5, 
        exploration_noise: Optional[Any]=None, 
        lam: float=0.1, 
        discount: float=0.99, 
        max_action: float=1.0, 
        device: Union[str, torch.device]="cpu"
    ) -> None:
        super().__init__()
        
        self.actor = actor
        self.actor_target = make_target(actor)
        self.critic = critic
        self.critic_target = make_target(critic)
        self.encoder = encoder
        self.fixed_encoder = make_target(encoder)
        self.fixed_encoder_target = make_target(encoder)

        self._offline = offline
        if not self._offline:
            self.checkpoint_actor = make_target(actor)
            self.checkpoint_encoder = make_target(encoder)

        self._actor_update_interval = actor_update_interval
        self._target_update_interval = target_update_interval
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._exploration_noise = exploration_noise
        self._lam = 0.0 if not self._offline else lam
        self._discount = discount
        self._max_action = max_action
        self._max_target_q = 0
        self._min_target_q = 0
        self._max_target_q_uptodate = -1e8
        self._min_target_q_uptodate = 1e8
        self._step = 0
        
        self.to(device)
        
    def configure_optimizers(
        self, 
        actor_lr: float, 
        critic_lr: float, 
        encoder_lr: float
    ) -> None:
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        
    @torch.no_grad()
    def select_action(
        self, 
        obs: np.ndarray, 
        use_checkpoint: bool=False, 
        deterministic: bool=False
    ) -> np.array:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        if use_checkpoint:
            zs = self.checkpoint_encoder.zs(obs)
            action = self.checkpoint_actor.sample(obs, zs, deterministic)[0].squeeze(0).cpu().numpy()
        else:
            zs = self.fixed_encoder.zs(obs)
            action = self.actor.sample(obs, zs, deterministic)[0].squeeze(0).cpu().numpy()
        if not deterministic and self._exploration_noise is not None:
            action = np.clip(action + self._exploration_noise(action.shape), -self._max_action, self._max_action)
        return action
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        metrics = {}
        self._step += 1
        
        ###### update encoder ######
        encoder_metrics = self.update_encoder(batch)
        metrics.update(encoder_metrics)
        
        ###### update critic #######
        critic_metrics, priority = self.update_critic(batch)
        metrics.update(critic_metrics)

        ###### update actor ########
        if self._step % self._actor_update_interval == 0:
            actor_metrics = self.update_actor(batch)
        else:
            actor_metrics = {}
        metrics.update(actor_metrics)

        if self._step % self._target_update_interval == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            self._max_target_q = self._max_target_q_uptodate
            self._min_target_q = self._min_target_q_uptodate
            
        return metrics, priority

    def update_encoder(self, batch):
        obss, actions, next_obss = itemgetter("observations", "actions", "next_observations")(batch)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_obss)
        zs = self.encoder.zs(obss)
        zsa_pred = self.encoder.zsa(zs, actions)
        encoder_loss = torch.nn.functional.mse_loss(zsa_pred, next_zs)
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        self.encoder_optim.step()
        return {"loss/encoder_loss": encoder_loss.item()}

    def update_critic(self, batch):
        obss, actions, next_obss, rewards, terminals = \
            itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        with torch.no_grad():
            fixed_target_next_zs = self.fixed_encoder_target.zs(next_obss)
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_target.sample(next_obss, fixed_target_next_zs)[0] + noise).clamp(-self._max_action, self._max_action)
            fixed_target_next_zsa = self.fixed_encoder_target.zsa(fixed_target_next_zs, next_actions)
            q_target = self.critic_target(next_obss, next_actions, fixed_target_next_zsa, fixed_target_next_zs).min(0)[0]
            q_target = rewards + self._discount * (1-terminals) * q_target.clamp(self._min_target_q, self._max_target_q)
            # track max/min target
            self._max_target_q_uptodate = max(self._max_target_q_uptodate, q_target.max().item())
            self._min_target_q_uptodate = min(self._min_target_q_uptodate, q_target.min().item())
            # inference for current obs/action embedding
            fixed_zs = self.fixed_encoder.zs(obss)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actions)
        q_pred = self.critic(obss, actions, fixed_zsa, fixed_zs)
        td = (q_target - q_pred).abs()
        critic_loss = torch.where(td < 1.0, 0.5 * td.pow(2), td).sum(0).mean()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        priority = td.max(dim=0)[0].detach().squeeze(-1).cpu().numpy()
        return {
            "loss/critic_loss": critic_loss.item(), 
            "misc/q_pred": q_pred.mean().item(), 
            "misc/max_q_uptodate": self._max_target_q_uptodate
        }, priority
        
    def update_actor(self, batch):
        obss, actions = itemgetter("observations", "actions")(batch)
        fixed_zs = self.fixed_encoder.zs(obss)
        new_actions = self.actor.sample(obss, fixed_zs)[0]
        fixed_zsa = self.fixed_encoder.zsa(fixed_zs, new_actions)
        q = self.critic(obss, new_actions, fixed_zsa, fixed_zs)

        actor_loss = - q.mean()
        if self._offline:
            bc_loss = q.abs().mean().detach() * torch.nn.functional.mse_loss(new_actions, actions)
            actor_loss += self._lam * bc_loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return {
            "loss/actor_loss": actor_loss.item(), 
            "loss/bc_loss": bc_loss.item() if self._offline else 0.0, 
        }

        