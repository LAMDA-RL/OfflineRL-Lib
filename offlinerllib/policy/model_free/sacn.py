from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy.model_free.sac import SACPolicy
from offlinerllib.utils.misc import convert_to_tensor


class SACNPolicy(SACPolicy):
    def __init__(
        self, 
        actor: BaseActor, 
        critic: Critic, 
        actor_optim: nn.Module, 
        critic_optim: nn.Module, 
        tau: float = 0.005, 
        discount: float = 0.99, 
        alpha: Union[float, Tuple[float, float]] = 0.2, 
        do_reverse_update: bool = False, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__(actor, critic, actor_optim, critic_optim, tau, discount, alpha, device)
        self.do_reverse_update = do_reverse_update
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.do_reverse_update:
            return self.reverse_update(batch)
        else:
            return super().update(batch)
    
    def reverse_update(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        metrics = {}
        obss = batch["observations"]
        
        # alpha update
        if self._is_auto_alpha:
            alpha_loss = self._alpha_loss(obss)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.exp().detach()
        else:
            alpha_loss = 0
        metrics["misc/alpha"] = self._alpha.item()
        
        # actor update
        actor_loss, actor_loss_metrics = self._actor_loss(batch)
        metrics.update(actor_loss_metrics)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # critic update
        critic_loss, critic_loss_metrics = self._critic_loss(batch)
        metrics.update(critic_loss_metrics)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._sync_weight()
        
        metrics.update({
            "loss/actor": actor_loss.item(), 
            "loss/critic": critic_loss.item(), 
            "loss/alpha": alpha_loss
        })
        return metrics
 