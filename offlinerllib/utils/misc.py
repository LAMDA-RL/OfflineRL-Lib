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

def merge_dict(dicts: List[Dict[str, torch.Tensor] | Dict[str, float]], is_tensor: bool=False) -> Dict[str, float]:
    result = {}
    for dict in dicts:
        result.update(dict)
    if is_tensor:
        result = {key:value.item() for (key, value) in result.items()}
    return result
