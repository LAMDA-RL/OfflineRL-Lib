from operator import itemgetter
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy.model_free.td3 import TD3Policy
from offlinerllib.utils.misc import convert_to_tensor, make_target


class TD3BCPolicy(TD3Policy):
    """
    TD3 with Behaviour Cloning <Ref: https://arxiv.org/pdf/2106.06860.pdf>
    """
    def __init__(
        self, 
        actor: BaseActor, 
        critic: Critic, 
        alpha: float = 0.2, 
        actor_update_interval: int = 2, 
        policy_noise: float = 0.2, 
        noise_clip: float = 0.5, 
        tau: float = 0.005, 
        discount: float = 0.99, 
        max_action: float = 1.0, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__(
            actor=actor, 
            critic=critic, 
            actor_update_interval=actor_update_interval, 
            policy_noise=policy_noise, 
            noise_clip=noise_clip, 
            exploration_noise=None, 
            tau=tau, 
            discount=discount, 
            max_action=max_action, 
            device=device
        )
        self.alpha = alpha

    def configure_optimizers(self, actor_lr, critic_lr):
        return super().configure_optimizers(actor_lr, critic_lr)
        
    def actor_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obss, actions = itemgetter("observations", "actions")(batch)
        new_actions, *_ = self.actor.sample(obss)
        new_q1 = self.critic(obss, new_actions)[0, ...]
        bc_loss = F.mse_loss(new_actions, actions)
        q_loss = -self.alpha / (new_q1.abs().mean().detach()) * new_q1.mean()
        total_loss = bc_loss + q_loss
        return q_loss+bc_loss, {
            "loss/actor_bc_loss": bc_loss.item(), 
            "loss/actor_q_loss": q_loss.item(), 
            "loss/actor_total_loss": total_loss.item()
        }
        
        