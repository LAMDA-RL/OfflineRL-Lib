from copy import deepcopy
from operator import itemgetter
from typing import Dict, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.module.actor import DeterministicActor, GaussianActor
from offlinerllib.policy.model_free import SACPolicy
from offlinerllib.utils.functional import expectile_regression
from offlinerllib.utils.misc import convert_to_tensor, make_target


class AWACPolicy(SACPolicy):
    """
    Advantage Weighted Actor Critic <Ref: https://arxiv.org/abs/2006.09359>
    """
    
    def __init__(
        self, 
        actor: nn.Module, 
        critic: nn.Module, 
        aw_lambda: float = 1.0, 
        discount: float = 0.99, 
        tau: float = 5e-3, 
        exp_adv_max: float = 100.0, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__(
            actor=actor, 
            critic=critic, 
            tau=tau, 
            discount=discount, 
            alpha=0.0, 
            device=device
        )
        self._exp_adv_max = exp_adv_max
        self._aw_lambda = aw_lambda

    def configure_optimizers(self, actor_lr, critic_lr):
        return super().configure_optimizers(actor_lr, critic_lr)
        
    def _actor_loss(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obss, actions = itemgetter("observations", "actions")(batch)
        with torch.no_grad():
            baseline_actions, *_ = self.actor.sample(obss)
            v = self.critic(obss, baseline_actions).min(0)[0]
            q = self.critic(obss, actions).min(0)[0]
            adv = ((q-v)/self._aw_lambda).exp().clamp(max=self._exp_adv_max)
        logprobs, *_ = self.actor.evaluate(obss, actions)
        actor_loss = -(adv * logprobs).mean()
        return actor_loss, {"loss/actor_loss": actor_loss.item()}
    
