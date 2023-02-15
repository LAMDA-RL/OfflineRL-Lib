from typing import Optional, Any, Dict, Sequence, Union

from torch.distributions import Normal

import torch
import math


class TanhNormal(Normal):
    def __init__(self, 
                 loc: torch.Tensor, 
                 scale: torch.Tensor, 
                 epsilon: float=1e-6
                 ):
        super().__init__(loc, scale)
        self.epsilon = epsilon
        
    def log_prob(self, 
                 value: torch.Tensor, 
                 pre_tanh_value: bool=False, 
                 ):
        if not pre_tanh_value:
            pre_value = torch.clip(value, -1.0+1e-6, 1.0-1e-6)
            pre_value = 0.5 * (pre_value.log1p() - (-pre_value).log1p())
        else:
            pre_value = value
            value = torch.tanh(pre_value)
        return super().log_prob(pre_value) - torch.log(1 - value.pow(2) + self.epsilon)
    
    def sample(self, sample_shape: Union[Sequence[int], int]=torch.Size([])):
        z = super().sample(sample_shape)
        return torch.tanh(z)
    
    def rsample(self, sample_shape: Union[Sequence[int], int]=torch.Size([])):
        z = super().rsample(sample_shape)
        return torch.tanh(z)
    
    def entropy(self):
        return super().entropy()
    
    @property
    def tanh_mean(self):
        return torch.tanh(self.mean)
    
    