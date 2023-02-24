from typing import Optional, Any, Union
import torch
import torch.nn as nn

from offlinerllib.policy import BasePolicy
from offlinerllib.module.net.attention import Transformer

class DecisionTransformerPolicy(BasePolicy):
    def __init__(
        self, 
        dt: Transformer, 
        dt_optim: torch.optim, 
        state_dim: int, 
        action_dim: int, 
        seq_len: int, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        
        self.dt = dt
        self.dt_optim = dt_optim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        
        self.to(device)

    @torch.no_grad()
    def select_action(self, states, actions, returns_to_go, timesteps, **kwargs):
        
        states = torch.from_numpy(states).float().reshape(1, -1, self.state_dim)[:, -self.seq_len:]
        actions = torch.from_numpy(actions).float().reshape(1, -1, self.action_dim)[:, -self.seq_len:]
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1, -1, 1)[:, -self.seq_len:]
        timesteps = torch.from_numpy(timesteps).float().reshape(1, -1)[:, -self.seq_len:]
        
        B, L, *_ = states.shape
        if self.seq_len > L:
            states = torch.cat([torch.zeros(B, self.seq_len-L, self.state_dim), states], dim=1)
            actions = torch.cat([torch.zeros(B, self.seq_len-L, self.action_dim), actions], dim=1)
            returns_to_go = torch.cat([torch.zeros(B, self.seq_len-L, 1), returns_to_go], dim=1)
            timesteps = torch.cat([torch.zeros(B, self.seq_len-L, 1), timesteps], dim=1)
            key_padding_mask = torch.cat([torch.ones(B, self.seq_len-L), torch.zeros(B, L)], dim=1).bool()
        else:
            key_padding_mask = torch.zeros(B, self.seq_len).bool()
        states, actions, returns_to_go, timesteps, key_padding_mask = \
            states.to(self.device), actions.to(self.device), returns_to_go.to(self.device), timesteps.to(self.device), key_padding_mask.to(self.device)
        
        action_pred = self.dt(
            states=states, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            key_padding_mask=key_padding_mask
        )
        
        return action_pred[0, -1].squeeze().cpu().numpy()
            
        
        
        
        

        
        
        
        
        