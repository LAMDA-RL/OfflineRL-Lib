import torch
import torch.nn as nn

from offlinerllib.module.net.attention.positional_encoding import BasePosEncoding


class DecayParameter(nn.Parameter):
    pass

class NoDecayParameter(nn.Parameter):
    pass


class BaseTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def configure_params(self):
        # https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215
        # However in this method we are not returning an Optimizer, but the parameter groups which
        # need / needn't weight decay, to support for more flexible downstream application
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, BasePosEncoding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif isinstance(p, DecayParameter):
                    decay.add(fpn)
                elif isinstance(p, NoDecayParameter):
                    no_decay.add(fpn)
                
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        return [param_dict[pn] for pn in sorted(list(decay))], [param_dict[pn] for pn in sorted(list(no_decay))]

