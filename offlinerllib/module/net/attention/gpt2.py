from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.base import BaseTransformer
from offlinerllib.module.net.attention.positional_encoding import PositionalEmbedding, SinusoidEncoding, ZeroEncoding


class GPTBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        backbone_dim: Optional[int]=None, 
    ) -> None:
        super().__init__()
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=attention_dropout, 
            batch_first=True
        )
        self.drop = nn.Dropout(residual_dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim), 
            nn.GELU(), 
            nn.Linear(backbone_dim, embed_dim), 
            nn.Dropout(residual_dropout) if residual_dropout else nn.Identity()
        )
    
    def forward(
        self, 
        input: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None
    ):
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
            
        residual = input
        input = self.ln1(input)
        attn_output = self.attention(
            query=input, 
            key=input, 
            value=input, 
            need_weights=False, 
            attn_mask=attention_mask, 
            key_padding_mask=key_padding_mask
        )[0]
        residual = residual + self.drop(attn_output) # this is because pytorch MHV don't do dropout after final projection
        residual = residual + self.ff(self.ln2(residual))
        return residual
        
        
class GPT2(BaseTransformer):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        num_heads: int, 
        causal: bool=True, 
        attention_dropout: Optional[float]=0.1, 
        residual_dropout: Optional[float]=0.1, 
        embed_dropout: Optional[float]=0.1, 
        pos_encoding: str="sinusoid", 
        pos_len: Optional[int]=None
    ) -> None:
        super().__init__()
        self.input_embed = nn.Linear(input_dim, embed_dim)
        pos_len = pos_len or 4096
        if pos_encoding == "sinusoid":
            self.pos_embed = SinusoidEncoding(embed_dim, pos_len)
        elif pos_encoding == "embedding":
            self.pos_embed = PositionalEmbedding(embed_dim, pos_len)
        elif pos_encoding == "none":
            self.pos_embed = ZeroEncoding(embed_dim, pos_len)
        self.embed_dropout = nn.Dropout(embed_dropout) if embed_dropout else nn.Identity()
        self.out_ln = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            GPTBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                attention_dropout=attention_dropout, 
                residual_dropout=residual_dropout
            ) for _ in range(num_layers)
        ])
        
        self.causal = causal
        
    def forward(
        self, 
        inputs: torch.Tensor, 
        timesteps: Optional[torch.Tensor]=None, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
        do_embedding: bool=True
    ):
        B, L, *_ = inputs.shape
        if self.causal:
            mask = ~torch.tril(torch.ones([L, L])).to(torch.bool).to(inputs.device)
        else:
            mask = torch.zeros([L, L]).to(torch.bool).to(inputs.device)
        if attention_mask is not None:
            mask = torch.bitwise_or(attention_mask.to(torch.bool), mask)
        
        if do_embedding:
            # do tokenize inside ?
            inputs = self.input_embed(inputs)
            if timesteps is None:
                timesteps = torch.arange(L).repeat(B, 1).to(inputs.device)
            inputs = inputs + self.pos_embed(timesteps)
        inputs = self.embed_dropout(inputs)
        for i, block in enumerate(self.blocks):
            inputs = block(inputs, attention_mask=mask, key_padding_mask=key_padding_mask)
        inputs = self.out_ln(inputs)
        return inputs

