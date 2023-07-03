import math
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.distributions import Normal


class TanhNormal(Normal):
    def __init__(self, 
                 loc: torch.Tensor, 
                 scale: torch.Tensor, 
                 ):
        super().__init__(loc, scale)
        self.epsilon = np.finfo(np.float32).eps.item()
        
    def log_prob(self, 
                 value: torch.Tensor, 
                 pre_tanh_value: bool=False, 
                 ):
        if not pre_tanh_value:
            pre_value = torch.clip(value, -1.0+self.epsilon, 1.0-self.epsilon)
            pre_value = 0.5 * (pre_value.log1p() - (-pre_value).log1p())
        else:
            pre_value = value
            value = torch.tanh(pre_value)
        return super().log_prob(pre_value) - 2*(math.log(2.0) - pre_value - torch.nn.functional.softplus(-2 * pre_value))
    
    def sample(self, sample_shape: Union[Sequence[int], int]=torch.Size([]), return_raw: bool=False):
        z = super().sample(sample_shape)
        return (torch.tanh(z), z) if return_raw else torch.tanh(z)
    
    def rsample(self, sample_shape: Union[Sequence[int], int]=torch.Size([]), return_raw: bool=False):
        z = super().rsample(sample_shape)
        return (torch.tanh(z), z) if return_raw else torch.tanh(z)
    
    def entropy(self):
        return super().entropy()
    
    @property
    def tanh_mean(self):
        return torch.tanh(self.mean)
    
    