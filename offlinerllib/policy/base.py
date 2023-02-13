import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Union, Any

class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    
    def select_action(self, obs: np.ndarray, *args, **kwargs):
        raise NotImplementedError
    
    def to(self, device):
        self.device = device
        super().to(device)
        return self