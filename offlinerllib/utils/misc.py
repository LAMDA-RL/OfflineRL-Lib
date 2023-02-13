import copy
import torch.nn as nn

def make_target(m: nn.Module) -> nn.Module:
    target = copy.deepcopy(m)
    target.requires_grad_(False)
    target.eval()
    return target