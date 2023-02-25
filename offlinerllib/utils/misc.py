import copy
import inspect
from typing import Any, Dict, List

import torch
import torch.nn as nn


def make_target(m: nn.Module) -> nn.Module:
    target = copy.deepcopy(m)
    target.requires_grad_(False)
    target.eval()
    return target

def convert_to_tensor(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return torch.from_numpy(obj).to(device)

def get_attributes(obj) -> Dict[str, Any]:
    return dict(inspect.getmembers(obj))

