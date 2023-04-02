from typing import Dict, Optional, Any, Sequence, Union, Callable, Type
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np

from offlinerllib.utils.distributions import TanhNormal
from offlinerllib.module.net.mlp import MLP, EnsembleMLP

ModuleType = Type[nn.Module]

class Critic(nn.Module):
    """
    A vanilla critic module, which can be used as Q(s, a) or V(s). 
    
    Notes
    -----
    All critics creates an extra post-processing module which maps the output of `backend` to
      the real final output. You can pass in any arguments for `MLP` or `EnsembleMLP` to 
      further customize the post-processing module. This is useful when you hope to, for example, 
      create an ensemble-style critic: just designating `ensemble_size`>1 when instantiaing.
    
    Parameters
    ----------
    backend :  The preprocessing backend of the critic, which is used to extract vectorized features from the raw input. 
    input_dim :  The dimensions of input (the output of backend module). 
    output_dim :  The dimension of critic's output. 
    device :  The device which the model runs on. Default is cpu. 
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self, 
        backend: nn.Module, 
        input_dim: int, 
        output_dim: int=1, 
        device: Union[str, int, torch.device] = "cpu", 
        *, 
        ensemble_size: int=1, 
        hidden_dims: Sequence[int] = [], 
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
        dropout: Optional[Union[float, Sequence[float]]] = None, 
        share_hidden_layer: Union[Sequence[bool], bool] = False, 
    ) -> None:
        super().__init__()
        self.critic_type = "Critic"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.ensemble_size = ensemble_size
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if ensemble_size == 1:
            self.output_layer = MLP(
                input_dim = input_dim, 
                output_dim = output_dim, 
                hidden_dims = hidden_dims, 
                norm_layer = norm_layer, 
                activation = activation, 
                dropout = dropout, 
                device = device
            )
        elif isinstance(ensemble_size, int) and ensemble_size > 1:
            self.output_layer = EnsembleMLP(
                input_dim = input_dim, 
                output_dim = output_dim, 
                hidden_dims = hidden_dims, 
                norm_layer = norm_layer, 
                activation = activation, 
                dropout = dropout, 
                device = device, 
                ensemble_size = ensemble_size, 
                share_hidden_layer = share_hidden_layer
            )
        else:
            raise ValueError(f"ensemble size should be int >= 1.")
    
    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor]=None, *args, **kwargs) -> torch.Tensor:
        """Compute the Q-value (when action is given) or V-value (when action is None). 
        
        Parameters
        ----------
        obs :  The observation, should be torch.Tensor. 
        action :  The action, should be torch.Tensor. 
        
        Returns
        -------
        torch.Tensor :  Q(s, a) or V(s). 
        """
        if action is not None:
            obs = torch.cat([obs, action], dim=-1)
        return self.output_layer(self.backend(obs))


class DoubleCritic(nn.Module):
    """
    Double Critic module, which consists of two (or more) independent Critic modules, can be used to implement the popular Double-Q trick. 
    
    Notes
    -----
    1. All critics creates an extra post-processing module which maps the output of `backend` to
      the real final output. You can pass in any arguments for `MLP` or `EnsembleMLP` to 
      further customize the post-processing module. This is useful when you hope to, for example, 
      create an ensemble-style critic: just designating `ensemble_size`>1 when instantiaing.
    2. Except for DoubleCritic. As we are handling ensemble explicitly with `critic_num`, you should not 
      specify `ensemble_size` or `share_hidden_layer` for this module any more. 
    
    Parameters
    ----------
    backend :  The preprocessing backend of the critic, which is used to extract vectorized features from the raw input. 
    input_dim :  The dimensions of input (the output of backend module). 
    output_dim :  The dimension of critic's output. 
    critic_num :  The num of critics. Default is 2. 
    reduce :  A unary function which specifies how to aggregate the output values. Default is torch.min along the 0 dimension. 
    device :  The device which the model runs on. Default is cpu. 
    ***(any args of MLP)
    """
    _reduce_fn_ = {
        "min": lambda x: torch.min(x, dim=0)[0],
        "max": lambda x: torch.max(x, dim=0)[0], 
        "mean": lambda x: torch.mean(x, dim=0)
    }
    def __init__(
        self, 
        backend: nn.Module, 
        input_dim: int, 
        output_dim: int=1, 
        critic_num: int=2, 
        reduce: Union[str, Callable]="min", 
        device: Union[str, int, torch.device]="cpu", 
        *, 
        hidden_dims: Sequence[int] = [], 
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
        dropout: Optional[Union[float, Sequence[float]]] = None, 
    ) -> None:
        super().__init__()
        self.critic_type = "DoubleCritic"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.critic_num = critic_num
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.output_layer = EnsembleMLP(
            input_dim = input_dim, 
            output_dim = output_dim, 
            ensemble_size = critic_num, 
            hidden_dims = hidden_dims, 
            norm_layer = norm_layer, 
            activation = activation, 
            dropout = dropout, 
            share_hidden_layer = False, 
            device = device
        )
        
        if isinstance(reduce, str):
            self.reduce = self._reduce_fn_[reduce]
        else:
            self.reduce = reduce
        
    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor]=None, reduce: bool=True, *args, **kwargs) -> torch.Tensor:
        """Compute the Q-value (when action is given) or V-value (when action is None), and aggregate them with the pre-defined reduce method. 
        If `reduce` is False, then no aggregation will be performed. 
        
        Parameters
        ----------
        obs :  The observation, should be torch.Tensor. 
        action :  The action, should be torch.Tensor. 
        reduce :  Whether to aggregate the outputs or not. Default is True. 
        
        Returns
        -------
        torch.Tensor :  Q(s, a) or V(s). 
        """
        if action is not None:
            obs = torch.cat([obs, action], dim=-1)
        output = self.output_layer(self.backend(obs))
        if reduce: 
            return self.reduce(output)
        else:
            return output
        
class C51DQN(nn.Module):
    """
    Implementation of Categorical Deep Q-Network. arXiv:1707.06887.
    
    Notes
    -----
    1. All critics creates an extra post-processing module which maps the output of `backend` to
      the real final output. You can pass in any arguments for `MLP` or `EnsembleMLP` to 
      further customize the post-processing module. This is useful when you hope to, for example, 
      create an ensemble-style critic: just designating `ensemble_size`>1 when instantiaing.
    2. Except for C51DQN. We don't support ensemble linear for output layers, so don't specify `ensemble_size` or 
      `share_hidden_layer` for this module. 
    
    Parameters
    ----------
    backend :  The preprocessing backend of the critic, which is used to extract vectorized features from the raw input. 
    input_dim :  The dimensions of input (the output of backend module). 
    output_dim_adv :  The dimension of advantage branch.
    output_dim_value :  The dimension of baseline value branch. Default is 1. 
    num_atoms : The number of atoms in the support set of the value distribution. Default is 51.
    v_min : The value of the smallest atom in the support set. Default is 0.0.
    v_max : The value of the largest atom in the support set. Default is 200.0.
    device :  The device which the model runs on. Default is cpu. 
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self, 
        backend: nn.Module, 
        input_dim: int, 
        output_dim_adv: int, 
        output_dim_value: int=1, 
        num_atoms: int=51, 
        v_min: float=0.0, 
        v_max: float=200.0, 
        device: Union[str, int, torch.device]="cpu", 
        *, 
        hidden_dims: Union[int, Sequence[int]] = [], 
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
        dropout: Optional[Union[float, Sequence[float]]] = None, 
    ) -> None:
        super().__init__()
        self.actor_type = "C51Actor"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim_adv = output_dim_adv
        self.output_dim_value = output_dim_value
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dims = hidden_dims.copy()
        self.device = device
        
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.adv_output_layer = MLP(
            input_dim = input_dim, 
            output_dim = output_dim_adv*num_atoms, 
            hidden_dims = hidden_dims, 
            norm_layer = norm_layer, 
            activation = activation, 
            dropout = dropout, 
            device = device, 
        )
        self.value_output_layer = MLP(
            input_dim = input_dim, 
            output_dim = output_dim_value*num_atoms, 
            hidden_dims = hidden_dims, 
            norm_layer = norm_layer, 
            activation = activation, 
            dropout = dropout, 
            device = device, 
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the value.
        
        Parameters
        ----------
        obs :  The observation, should be torch.Tensor. 
        
        Returns
        -------
        torch.Tensor :  V(s). 
        """
        dist = self.dist(obs)
        q = torch.sum(dist * self.support, dim=2)
        return q
        
    def dist(self, state: torch.Tensor):
        o_backend = self.backend(state)
        o_adv = self.adv_output_layer(o_backend).view(-1, self.output_dim_adv, self.num_atoms)
        o_value = self.value_output_layer(o_backend).view(-1, self.output_dim_value, self.num_atoms)
        q_atoms = o_value + o_adv - o_adv.mean(dim=-2, keepdim=True)
        return nn.functional.softmax(q_atoms, dim=-1).clamp(min=1e-3)
        
            