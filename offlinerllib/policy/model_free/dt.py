from operator import itemgetter
from typing import Any, Dict, Optional, Union

import torch

from offlinerllib.module.net.attention.dt import DecisionTransformer
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor
from offlinerllib.module.actor import (
    SquashedDeterministicActor, 
    SquashedGaussianActor, 
    CategoricalActor
)


class DecisionTransformerPolicy(BasePolicy):
    """
    Decision Transformer: Reinforcement Learning via Sequence Modeling <Ref: https://arxiv.org/abs/2106.01345>
    """
    def __init__(
        self, 
        dt: DecisionTransformer, 
        state_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        seq_len: int, 
        episode_len: int, 
        use_abs_timestep: bool=True, 
        policy_type: str="deterministic", 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.episode_len = episode_len
        self.use_abs_timestep = use_abs_timestep
        self.policy_type = policy_type
        
        if policy_type == "deterministic":
            self.policy_head = SquashedDeterministicActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=action_dim
            )
        elif policy_type == "stochastic":
            self.policy_head = SquashedGaussianActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=action_dim, 
                reparameterize=False, 
            )
        elif policy_type == "categorical":
            self.policy_head = CategoricalActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=action_dim
            )
        self.to(device)

    def configure_optimizers(self, lr, weight_decay, betas, warmup_steps):
        decay, no_decay = self.dt.configure_params()
        self.dt_optim = torch.optim.AdamW([
            {"params": [*decay, *self.policy_head.parameters()], "weight_decay": weight_decay}, 
            {"params": no_decay, "weight_decay": 0.0}
        ], lr=lr, betas=betas)
        self.dt_optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dt_optim, lambda step: min((step+1)/warmup_steps, 1))

    @torch.no_grad()
    def select_action(self, states, actions, returns_to_go, timesteps, deterministic=False, **kwargs):
        states = torch.from_numpy(states).float().reshape(1, -1, self.state_dim)[:, -self.seq_len:].to(self.device)
        actions = torch.from_numpy(actions).float().reshape(1, -1, self.action_dim)[:, -self.seq_len:].to(self.device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1, -1, 1)[:, -self.seq_len:].to(self.device)
        timesteps = torch.from_numpy(timesteps).reshape(1, -1)[:, -self.seq_len:].to(self.device)
        
        B, L, *_ = states.shape
        out = self.dt(
            states=states, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps if self.use_abs_timestep else None, 
            attention_mask=None, 
            key_padding_mask=None
        )
        action_pred = self.policy_head.sample(out[:, 1::3], deterministic=deterministic)[0]
        return action_pred[0, L-1].squeeze().cpu().numpy() 
    
    def update(self, batch: Dict[str, Any], clip_grad: Optional[float]=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        obss, actions, returns_to_go, timesteps, masks = \
            itemgetter("observations", "actions", "returns", "timesteps", "masks")(batch)
        key_padding_mask = ~masks.to(torch.bool)
        
        out = self.dt(
            states=obss, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps if self.use_abs_timestep else None, 
            attention_mask=None,    # DT is causal and will handle causal masks itself
            key_padding_mask=key_padding_mask
        )
        if isinstance(self.policy_head, SquashedDeterministicActor):
            actor_loss = torch.nn.functional.mse_loss(
                self.policy_head.sample(out[:, 1::3])[0], 
                actions.detach(), 
                reduction="none"
            )
        elif isinstance(self.policy_head, SquashedGaussianActor):
            actor_loss = self.policy_head.evaluate(
                out[:, 1::3], 
                actions.detach(), 
            )[0]
        elif isinstance(self.policy_head, CategoricalActor):
            actor_loss = self.policy_head.evaluate(
                out[:, 1::3], 
                actions.detach(),
                is_onehot_action=False
            )
        actor_loss = (actor_loss * masks.unsqueeze(-1)).mean()
        self.dt_optim.zero_grad()
        actor_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.dt.parameters(), clip_grad)
        self.dt_optim.step()
        self.dt_optim_scheduler.step()
        
        return {
            "loss/actor_loss": actor_loss.item(), 
            "misc/learning_rate": self.dt_optim_scheduler.get_last_lr()[0]
        }
        
        