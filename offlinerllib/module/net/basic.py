from typing import List, Optional, Type, Any

import math
import torch
import torch.nn as nn

ModuleType = Type[nn.Module]

def miniblock(
    input_dim: int,
    output_dim: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
    dropout: Optional[ModuleType] = None, 
    linear_layer: ModuleType = nn.Linear,
    *args, 
    **kwargs
) -> List[nn.Module]:
    """
    Construct a miniblock with given input and output. It is possible to specify norm layer, activation, and dropout for the constructed miniblock.
    
    Parameters
    ----------
    input_dim :  Number of input features..
    output_dim :  Number of output features. Default is 0.
    norm_layer :  Module to use for normalization. When not specified or set to True, nn.LayerNorm is used.
    activation :  Module to use for activation. When not specified or set to True, nn.ReLU is used.
    dropout :  Dropout rate. Default is None.
    linear_layer :  Module to use for linear layer. Default is nn.Linear.
    
    Returns
    -------
    List of modules for miniblock.
    """

    layers: List[nn.Module] = [linear_layer(input_dim, output_dim, *args, **kwargs)]
    if norm_layer is not None:
        if isinstance(norm_layer, nn.Module):
            layers += [norm_layer(output_dim)]
        else:
            layers += [nn.LayerNorm(output_dim)]
    if activation is not None:
        layers += [activation()]
    if dropout is not None and dropout > 0:
        layers += [nn.Dropout(dropout)]
    return layers


class EnsembleLinear(nn.Module):
    """
    An linear module for concurrent forwarding, which can be used for ensemble purpose.
    
    Parameters
    ----------
    in_features :  Number of input features. 
    out_features :  Number of output features. 
    ensemble_size :  Ensemble size. Default is 1.
    bias :  Whether to add bias or not. Default is True.
    device :  Device to use for parameters.
    dtype :  Data type to use for parameter.
    """
    def __init__(
        self,
        in_features, 
        out_features, 
        ensemble_size: int = 1,
        share_input: bool=True, 
        bias: bool = True, 
        device: Optional[Any] = None, 
        dtype: Optional[Any] = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.share_input = share_input
        self.add_bias = bias
        self.register_parameter("weight", torch.nn.Parameter(torch.empty([in_features, out_features, ensemble_size], **factory_kwargs)))
        if bias:
            self.register_parameter("bias", torch.nn.Parameter(torch.empty([out_features, ensemble_size], **factory_kwargs)))
        else:
            self.register_buffer("bias", torch.zeros([out_features, ensemble_size], **factory_kwargs))
        self.reset_parameters()
        
    def reset_parameters(self):
        for i in range(self.ensemble_size):
            torch.nn.init.kaiming_uniform_(self.weight[..., i], a=math.sqrt(5))
        if self.add_bias:
            for i in range(self.ensemble_size):
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[..., i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias[..., i], -bound, bound)
        
    def forward(self, input: torch.Tensor):
        if self.share_input:
            res = torch.einsum('...j,jkb->...kb', input, self.weight) + self.bias
        else:
            res = torch.einsum('b...j,jkb->...kb', input, self.weight) + self.bias
        return torch.einsum('...b->b...', res)
    
    def __repr__(self):
        return f"EnsembleLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.add_bias})"


class NoisyLinear(nn.Linear):
    """
    An linear module which supports for noisy parameters.
    
    Parameters
    ----------
    in_features :  Number of input features.
    out_features :  Number of output features.
    std_init :  Standard deviation of the weight and noise initialization.
    bias :  Whether to add bias. 
    device :  Device to use for parameters.
    dtype :  Data type to use for parameter.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float=0.5, 
        bias: bool=True, 
        device=None, 
        dtype=None
    ):
        super().__init__(in_features, out_features, bias, device, dtype) # this will do the initialization

        self.std_init = std_init
        self.register_parameter("weight_std", torch.nn.Parameter(torch.empty(out_features, in_features)))
        self.weight_std.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.register_buffer("weight_noise", torch.empty_like(self.weight))
        if bias:
            self.register_parameter("bias_std", torch.nn.Parameter(torch.empty(out_features)))
            self.bias_std.data.fill_(self.std_init / math.sqrt(self.out_features))
            self.register_buffer("bias_noise", torch.empty_like(self.bias))
        else:
            self.register_parameter("bias_std", None)
            self.register_buffer("bias_noise", None)
        
        self.reset_noise()

    @staticmethod
    def scaled_noise(size: int):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor, reset_noise=False):
        if self.training:
            if reset_noise:
                self.reset_noise()
            if self.bias is not None:
                return torch.nn.functional.linear(
                        x, 
                        self.weight + self.weight_std * self.weight_noise, 
                        self.bias + self.bias_std * self.bias_noise
                    )
            else:
                return torch.nn.functional.linear(
                    x, 
                    self.weight + self.weight_std * self.weight_noise, 
                    None
                )
        else:
            return torch.nn.functional.linear(x, self.weight, self.bias)
        
    def reset_noise(self):
        """
        Reset the noise to the noise matrix .
        """
        device = self.weight.data.device
        epsilon_in = self.scaled_noise(self.in_features)
        epsilon_out = self.scaled_noise(self.out_features)
        self.weight_noise.data = torch.matmul(
            torch.unsqueeze(epsilon_out, -1), other=torch.unsqueeze(epsilon_in, 0)
        ).to(device)
        if self.bias is not None:
            self.bias_noise.data = epsilon_out.to(device)
        
        