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
        seq_len: int, 
        num_heads: int, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        backbone_dim: Optional[int]=None, 
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        self.attention = Attention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
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
    
    def forward(
        self, 
        input: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None
    ):
        B, L, *shape = input.shape
        if attention_mask is None:
            attention_mask = self.mask[:L, :L]
        else:
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
        residual = residual + attn_output
        residual = residual + self.ff(self.ln2(residual))
        return residual
        
        
class Transformer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        num_heads: int, 
        causal: bool=False, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        embed_dropout: Optional[float]=None, 
        position_dim: Optional[int]=None
    ) -> None:
        super().__init__()
        self.input_embed = nn.Linear(input_dim, embed_dim)
        position_dim = position_dim or seq_len
        self.pos_embed = nn.Embedding(position_dim, embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout) if embed_dropout else nn.Identity()
        self.out_ln = nn.LayerNorm(embed_dim)
        # self.embed_ln = nn.LayerNorm(embed_dim)   # check: whether or not add layer norm before embed dropout
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, 
                seq_len=seq_len, 
                num_heads=num_heads, 
                attention_dropout=attention_dropout, 
                residual_dropout=residual_dropout
            ) for _ in range(num_layers)
        ])
        
        self.causal = causal
        if causal:
            self.register_buffer("causal_mask", ~torch.tril(torch.ones([seq_len, seq_len])).to(torch.bool))
        else:
            self.register_buffer("causal_mask", torch.zeros([seq_len, seq_len]).to(torch.bool))
        
    def forward(
        self, 
        inputs: torch.Tensor, 
        timesteps: Optional[torch.Tensor]=None, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
        do_embedding: bool=True
    ):
        B, L, *_ = inputs.shape
        mask = self.causal_mask[:L, :L]
        # deal with the masks
        if attention_mask is not None:
            mask = torch.bitwise_or(attention_mask.to(torch.bool), mask)
        
        if do_embedding:
            # do tokenize inside ?
            inputs = self.input_embed(inputs)
            if timesteps is not None:
                inputs = inputs + self.pos_embed(timesteps)
        inputs = self.embed_dropout(inputs)
        for i, block in enumerate(self.blocks):
            inputs = block(inputs, attention_mask=mask, key_padding_mask=key_padding_mask)
        inputs = self.out_ln(inputs)
        return inputs
    
    
class DecisionTransformer(Transformer):
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        episode_len: int=1000, 
        num_heads: int=4, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        embed_dropout: Optional[float]=None, 
    ) -> None:
        super().__init__(
            input_dim=embed_dim, # actually not used
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            seq_len=3*seq_len, # (return_to_go, state, action)
            num_heads=num_heads, 
            causal=True, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            embed_dropout=embed_dropout, 
        )
        # we manually do the embeddings here
        self.pos_embed = nn.Embedding(episode_len + seq_len, embed_dim)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.ret_embed = nn.Linear(1, embed_dim)
        
        self.action_head = nn.Sequential(nn.Linear(embed_dim, action_dim), nn.Tanh())

    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        returns_to_go: torch.Tensor, 
        timesteps: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        B, L, *_ = states.shape
        time_embedding = self.pos_embed(timesteps)
        state_embedding = self.obs_embed(states) + time_embedding
        action_embedding = self.act_embed(actions) + time_embedding
        return_embedding = self.ret_embed(returns_to_go) + time_embedding
        
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask, key_padding_mask, key_padding_mask], dim=2).reshape(B, 3*L)
        
        stacked_input = torch.stack([return_embedding, state_embedding, action_embedding], dim=2).reshape(B, 3*L, state_embedding.shape[-1])
        out = super().forward(
            inputs=stacked_input, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )

        out = self.action_head(out[:, 1::3])
        return out    # (batch size, length, action_shape)

        