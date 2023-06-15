import torch
import torch.nn as nn
import torch.nn.functional as F

from offlinerllib.module.actor import BaseActor
from offlinerllib.module.net.mlp import EnsembleLinear

class AvgL1Norm(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, eps: float=1e-8):
        return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)
    
def avg_l1_norm(x: torch.Tensor, eps: float=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class TD7Encoder(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        activation=nn.ELU
    ) -> None:
        super().__init__()
        
        # state encoder
        self.zs_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),     
            nn.Linear(hidden_dim, hidden_dim), 
            activation(),  
            nn.Linear(hidden_dim, embedding_dim), 
            AvgL1Norm()
        )
        self.zsa_layers = nn.Sequential(
            nn.Linear(embedding_dim+action_dim, hidden_dim), 
            activation(), 
            nn.Linear(hidden_dim, hidden_dim), 
            activation(), 
            nn.Linear(hidden_dim, embedding_dim)
            # we don't add AvgL1Norm here because it is regressed towards normed zs
        )
        
    def zs(self, state: torch.Tensor):
        return self.zs_layers(state)

    def zsa(self, zs: torch.Tensor, action: torch.Tensor):
        out = torch.concat([zs, action], dim=-1)
        return self.zsa_layers(out)
    

class TD7Actor(BaseActor):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        activation=nn.ReLU
    ) -> None:
        super().__init__()
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            AvgL1Norm()
        )
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim+hidden_dim, hidden_dim), 
            activation(), 
            nn.Linear(hidden_dim, hidden_dim), 
            activation(), 
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor, zs: torch.Tensor):
        out = self.state_layers(state)
        out = torch.concat([out, zs], dim=-1)
        out = self.layers(out)
        return out
    
    def sample(self, state: torch.Tensor, zs: torch.Tensor, *args, **kwargs):
        return torch.tanh(self.forward(state, zs)), None, {}
    
    
class TD7Critic(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        embedding_dim: int, 
        hidden_dim: int,  
        critic_num: int=2, 
        activation=nn.ReLU, 
    ) -> None:
        super().__init__()
        self.critic_num = critic_num
        self.sa_layers = nn.Sequential(
            EnsembleLinear(state_dim+action_dim, hidden_dim, ensemble_size=critic_num), 
            AvgL1Norm()
        )
        self.layers = nn.Sequential(
            EnsembleLinear(2*embedding_dim+hidden_dim, hidden_dim, ensemble_size=critic_num),
            activation(), 
            EnsembleLinear(hidden_dim, hidden_dim, ensemble_size=critic_num), 
            activation(), 
            EnsembleLinear(hidden_dim, 1, ensemble_size=critic_num) 
        )

    def forward(
        self, 
        state: torch.Tensor,
        action: torch.Tensor, 
        zsa: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        out = torch.concat([state, action], dim=-1)
        out = self.sa_layers(out)
        out = torch.concat([
            out, 
            zsa.repeat([self.critic_num]+[1]*len(zsa.shape)), 
            zs.repeat([self.critic_num]+[1]*len(zs.shape))
        ], dim=-1)
        out = self.layers(out)
        return out
        
        