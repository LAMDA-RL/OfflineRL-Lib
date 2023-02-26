from operator import itemgetter
from typing import Any, Dict, Optional, Union

import torch

from offlinerllib.module.net.attention import Transformer
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor


class DecisionTransformerPolicy(BasePolicy):
    """
    Decision Transformer: Reinforcement Learning via Sequence Modeling <Ref: https://arxiv.org/abs/2106.01345>
    """
    def __init__(
        self, 
        dt: Transformer, 
        dt_optim: torch.optim, 
        state_dim: int, 
        action_dim: int, 
        seq_len: int, 
        episode_len: int, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.dt = dt
        self.dt_optim = dt_optim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.episode_len = episode_len
        
        self.to(device)

    @torch.no_grad()
    def select_action(self, states, actions, returns_to_go, timesteps, **kwargs):
        states = torch.from_numpy(states).float().reshape(1, -1, self.state_dim)[:, -self.seq_len:]
        actions = torch.from_numpy(actions).float().reshape(1, -1, self.action_dim)[:, -self.seq_len:]
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1, -1, 1)[:, -self.seq_len:]
        timesteps = torch.from_numpy(timesteps).reshape(1, -1)[:, -self.seq_len:]
        
        B, L, *_ = states.shape
        if self.seq_len > L:
            states = torch.cat([torch.zeros(B, self.seq_len-L, self.state_dim), states], dim=1)
            actions = torch.cat([torch.zeros(B, self.seq_len-L, self.action_dim), actions], dim=1)
            returns_to_go = torch.cat([torch.zeros(B, self.seq_len-L, 1), returns_to_go], dim=1)
            timesteps = torch.cat([torch.zeros(B, self.seq_len-L, dtype=torch.int64), timesteps], dim=1)
        states, actions, returns_to_go, timesteps = \
            states.to(self.device), actions.to(self.device), returns_to_go.to(self.device), timesteps.to(self.device)
        
        action_pred = self.dt(
            states=states, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            attention_mask=None, 
            key_padding_mask=None   # we don't need to pass key_padding_mask as all the keys are valid
        )
        
        return action_pred[0, -1].squeeze().cpu().numpy()
    
    def update(self, batch: Dict[str, Any], clip_grad: Optional[float]=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        obss, actions, returns_to_go, timesteps, masks = \
            itemgetter("observations", "actions", "returns", "timesteps", "masks")(batch)
        key_padding_mask = ~masks.to(torch.bool)
        
        action_pred = self.dt(
            states=obss, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            attention_mask=None,    # DT is causal and will handle causal masks itself
            key_padding_mask=key_padding_mask
        )
        mse_loss = torch.nn.functional.mse_loss(action_pred, actions.detach(), reduction="none")
        mse_loss = (mse_loss * masks.unsqueeze(-1)).mean()
        self.dt_optim.zero_grad()
        mse_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.dt.parameters(), clip_grad)
        self.dt_optim.step()
        
        return {
            "loss/mse_loss": mse_loss.item(), 
        }
        
        