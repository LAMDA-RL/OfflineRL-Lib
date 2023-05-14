import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    
class NoDecayParameter(nn.Parameter):
    pass

class DecayParameter(nn.Parameter):
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
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, DecayParameter)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, PositionalEncoding, NoDecayParameter)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
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

