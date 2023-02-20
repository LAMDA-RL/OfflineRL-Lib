from turtle import back
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.basic import miniblock, EnsembleLinear

Attention: nn.Module = nn.MultiheadAttention

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        n_head: int, 
        seq_len: int, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        causal: bool=False, 
        backbone_dim: Optional[int]=None, 
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        self.attention = Attention(
            embed_dim=embed_dim, 
            num_heads=n_head, 
            dropout=attention_dropout, 
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim), 
            nn.GELU(), 
            nn.Linear(backbone_dim, embed_dim), 
            nn.Dropout(residual_dropout) if residual_dropout else nn.Identity()
        )
        if causal:
            self.register_buffer("mask", ~torch.tril(torch.ones([seq_len, seq_len])).to(torch.bool))
        else:
            self.register_buffer("mask", torch.zeros([seq_len, seq_len]).to(torch.bool))
    
    def forward(
        self, 
        input: torch.Tensor, 
        attention_mask: Optional[Any]=None, 
    ):
        B, L, *shape = input.shape
        mask = self.mask[:L, :L]
        if attention_mask is not None:
            mask += attention_mask
            
        residual = input
        input = self.ln1(input)
        attn_output = self.attention(
            query=input, 
            key=input, 
            value=input, 
            need_weights=False, 
            attn_mask=mask
        )[0]
        residual = residual + attn_output
        residual = residual + self.ff(self.ln2(residual))
        return residual
        
