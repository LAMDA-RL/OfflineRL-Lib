from operator import itemgetter
from typing import Dict, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target
from offlinerllib.utils.functional import expectile_regression


class PORPolicy(BasePolicy):
    """
    A Policy-Guided Imitation Approach for Offline Reinforcement Learning <Ref: https://arxiv.org/abs/2210.08323>
    """
    def __init__(
        self, 
        actor: nn.Module, 
        goal_actor: nn.Module, 
        behavior_goal_actor: nn.Module, 
        critic_v: nn.Module,
        variant: str="residual", 
        v_expectile: float=0.7, 
        alpha: float=10.0,
        discount: float=0.99, 
        tau: float=0.005, 
        exp_adv_max: float=100.0, 
        device: Union[str, torch.device]="cpu"
    ) -> None:
        super().__init__()
        self.actor = actor
        self.goal_actor = goal_actor
        self.behavior_goal_actor = behavior_goal_actor
        self.critic_v = critic_v
        self.critic_v_target = make_target(critic_v)
        if variant not in {"residual", "qlearning"}:
            raise ValueError("Variant of POR should be either residual or qlearning. ")
        self.variant = variant
        self._v_expectile = v_expectile
        self._discount = discount
        self._tau = tau
        self._alpha = alpha
        self._exp_adv_max = exp_adv_max
        self.to(device)
        
    def configure_optimizers(
        self, 
        actor_lr: float=3e-4, 
        critic_v_lr: float=3e-4, 
        actor_lr_scheduler_max_steps: int=1000000
    ) -> None:
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=critic_v_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.goal_actor_optim = torch.optim.Adam(self.goal_actor.parameters(), lr=actor_lr)
        self.behavior_goal_actor_optim = torch.optim.Adam(self.behavior_goal_actor.parameters(), lr=0.0001)
        self.actor_optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim, actor_lr_scheduler_max_steps)
        self.goal_actor_optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.goal_actor_optim, actor_lr_scheduler_max_steps)
    
    @torch.no_grad()
    def select_action(
        self, 
        obs: np.ndarray, 
        deterministic: bool=False, 
        *args, **kwargs
    ) -> np.ndarray:
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        goal, *_ = self.goal_actor.sample(obs, deterministic=deterministic)
        action, *_ = self.actor.sample(torch.concat([obs, goal], dim=-1), deterministic=deterministic)
        return action.squeeze().cpu().numpy()
    
    def update_critic_v(self, batch):
        obss, rewards, terminals, next_obss = \
            itemgetter("observations", "rewards", "terminals", "next_observations")(batch)
        with torch.no_grad():
            target_v = rewards + self._discount*(1 - terminals)*self.critic_v_target(next_obss)
        vs = self.critic_v(obss, reduce=False)
        v_loss = expectile_regression(vs, target_v, expectile=self._v_expectile).sum(0).mean()
        self.critic_v_optim.zero_grad()
        v_loss.backward()
        self.critic_v_optim.step()
        self._sync_weight()
        return target_v, {
            "loss/v_loss": v_loss.item(), 
            "misc/v_value": vs.mean().item()
        }
        
    def update_goal_actor_residual(self, batch, target_v):
        obss, next_obss = itemgetter("observations", "next_observations")(batch)
        with torch.no_grad():
            vs = self.critic_v(obss)
            advs = target_v - vs
            weights = (self._alpha * advs).exp().clamp(max=self._exp_adv_max)
        goal_neglogprobs = -self.goal_actor.evaluate(obss, next_obss)[0]
        goal_loss = (weights * goal_neglogprobs).mean()
        self.goal_actor_optim.zero_grad()
        goal_loss.backward()
        self.goal_actor_optim.step()
        self.goal_actor_optim_scheduler.step()
        return {
            "goal_residual/goal_loss": goal_loss.item(), 
            "goal_residual/advs_scale_raw": advs.abs().mean().item()
        }
        
    def update_goal_actor_qlearning(self, batch):
        obss = itemgetter("observations")(batch)
        goal_samples = self.goal_actor.sample(obss)[0]
        goal_loss1 = -self.critic_v(goal_samples)
        goal_loss2 = -self.behavior_goal_actor.evaluate(obss, goal_samples)[0].mean()
        lmbda = self._alpha / goal_loss1.abs().mean().detach()
        goal_loss = lmbda * goal_loss1.mean() + goal_loss2
        self.goal_actor_optim.zero_grad()
        goal_loss.backward()
        self.goal_actor_optim.step()
        self.goal_actor_optim_scheduler.step()
        return {
            "goal_qlearning/goal_loss_total": goal_loss.item(), 
            "goal_qlearning/goal_loss1": goal_loss1.mean().item(), 
            "goal_qlearning/goal_loss2": goal_loss2.item()
        }
    
    def update_actor(self, batch):
        obss, actions, next_obss = itemgetter("observations", "actions", "next_observations")(batch)
        actor_neglogprobs = -self.actor.evaluate(torch.concat([obss, next_obss], dim=-1), actions)[0]
        actor_loss = actor_neglogprobs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.actor_optim_scheduler.step()
        return {
            "loss/actor_loss": actor_loss.item()
        }
        
    def pretrain(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        obss, next_obss = itemgetter("observations", "next_observations")(batch)
        bc_loss = - self.behavior_goal_actor.evaluate(obss, next_obss)[0].mean()
        self.behavior_goal_actor_optim.zero_grad()
        bc_loss.backward()
        self.behavior_goal_actor_optim.step()
        return {
            "pretrain/behavior_goal_loss": bc_loss.item()
        }
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        metrics = dict()
        
        target_v, v_metrics = self.update_critic_v(batch)
        metrics.update(v_metrics)

        if self.variant == "residual":
            goal_metrics = self.update_goal_actor_residual(batch, target_v)
            metrics.update(goal_metrics)
        elif self.variant == "qlearning":
            goal_metrics = self.update_goal_actor_qlearning(batch)
            metrics.update(goal_metrics)
        
        actor_metrics = self.update_actor(batch)
        metrics.update(actor_metrics)
        
        return metrics

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)