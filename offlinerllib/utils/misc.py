import copy
import inspect
import torch
import torch.nn as nn
from typing import Any, List, Dict

def make_target(m: nn.Module) -> nn.Module:
    target = copy.deepcopy(m)
    target.requires_grad_(False)
    target.eval()
    return target

def get_attributes(obj) -> Dict[str, Any]:
    return dict(inspect.getmembers(obj))

