from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.base import BaseTransformer
from offlinerllib.module.net.attention.positional_encoding import PositionalEmbedding, SinusoidEncoding, ZeroEncoding


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        backbone_dim: Optional[int]=None, 
        dropout: Optional[float]=None, 
    ) -> None:
        super().__init__()
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        # in transformer we don't add dropout inside MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(backbone_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        input: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)

        residual = input
        input = self.attention(
            query=input, 
            key=input, 
            value=input, 
            need_weights=False, 
            attn_mask=attention_mask, 
            key_padding_mask=key_padding_mask
        )[0]
        residual = self.norm1(residual + self.dropout1(input))
        residual = self.norm2(residual + self.dropout2(self.ff(residual)))
        return residual
    

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        backbone_dim: Optional[int]=None, 
        dropout: Optional[float]=None
    ) -> None:
        super().__init__()
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.enc_dec_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(backbone_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self, 
        tgt: torch.Tensor, 
        enc_src: Optional[torch.Tensor]=None, 
        tgt_attention_mask: Optional[torch.Tensor]=None, 
        tgt_key_padding_mask: Optional[torch.Tensor]=None, 
        src_attention_mask: Optional[torch.Tensor]=None, 
        src_key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        # compute self.attention
        _x = tgt
        x = self.self_attention(
            query=tgt, 
            key=tgt, 
            value=tgt, 
            attn_mask=tgt_attention_mask, 
            key_padding_mask=tgt_key_padding_mask
        )[0]
        x = self.norm1(_x + self.dropout1(x))
        
        if enc_src is not None:
            # compute encoder-decoder attention
            _x = x
            x = self.enc_dec_attention(
                query=x, 
                key=enc_src, 
                value=enc_src, 
                attn_mask=src_attention_mask, 
                key_padding_mask=src_key_padding_mask
            )[0]
            x = self.norm2(_x + self.dropout2(x))
        
        # ffn
        _x = x
        x = self.norm3(_x + self.dropout3(self.ff(x)))
        return x


class TransformerEncoder(BaseTransformer):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        num_heads: int, 
        causal: bool=False, 
        embed_dropout: Optional[float]=None, 
        attention_dropout: Optional[float]=None, 
        pos_encoding: str="sinusoid", 
        pos_len: Optional[int]=None, 
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
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                dropout=attention_dropout
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
            inputs = self.input_embed(inputs)
            if timesteps is None:
                timesteps = torch.arange(L).repeat(B, 1).to(inputs.device)
            inputs = inputs + self.pos_embed(timesteps)
        inputs = self.embed_dropout(inputs)
        for i, block in enumerate(self.blocks):
            inputs = block(inputs, attention_mask=mask, key_padding_mask=key_padding_mask)
        return inputs
    

class TransformerDecoder(BaseTransformer):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        num_heads: int, 
        causal: bool=True, 
        embed_dropout: Optional[float]=None, 
        attention_dropout: Optional[float]=None, 
        pos_encoding: str="sinusoid", 
        pos_len: Optional[int]=None, 
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
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                dropout=attention_dropout, 
            ) for _ in range(num_layers)
        ])
        
        self.causal = causal
            
    def forward(
        self, 
        tgt: torch.Tensor, 
        enc_src: Optional[torch.Tensor]=None, 
        timesteps: Optional[torch.Tensor]=None, 
        tgt_attention_mask: Optional[torch.Tensor]=None, 
        tgt_key_padding_mask: Optional[torch.Tensor]=None, 
        src_attention_mask: Optional[torch.Tensor]=None, 
        src_key_padding_mask: Optional[torch.Tensor]=None, 
        do_embedding: bool=True
    ):
        B, L, *_ = tgt.shape
        if self.causal:
            tgt_mask = ~torch.tril(torch.ones([L, L])).to(torch.bool).to(tgt.device)
        else:
            tgt_mask = torch.zeros([L, L]).to(torch.bool).to(tgt.device)
        if tgt_attention_mask is not None:
            tgt_mask = torch.bitwise_or(tgt_attention_mask.to(torch.bool), tgt_mask)
        if do_embedding:
            tgt = self.input_embed(tgt)
            if timesteps is not None:
                timesteps = torch.arange(L).repeat(B, 1).to(tgt.device)
            tgt = tgt + self.pos_embed(timesteps)
        output = self.embed_dropout(tgt)
        for i, block in enumerate(self.blocks):
            output = block(
                tgt=output, 
                enc_src=enc_src, 
                tgt_attention_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask, 
                src_attention_mask=src_attention_mask, 
                src_key_padding_mask=src_key_padding_mask
            )
        return output
        

class Transformer(BaseTransformer):
    def __init__(
        self, 
        enc_input_dim: int, 
        dec_input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        num_heads: int, 
        embed_dropout: Optional[float]=None, 
        attention_dropout: Optional[float]=None, 
        pos_encoding: str="sinusoid", 
        pos_len: Optional[int]=None, 
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            input_dim=enc_input_dim, 
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            causal=False, 
            embed_dropout=embed_dropout, 
            attention_dropout=attention_dropout, 
            pos_encoding=pos_encoding, 
            pos_len=pos_len
        )
        self.decoder = TransformerDecoder(
            input_dim=dec_input_dim, 
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            num_heads=num_heads,
            causal=True, 
            embed_dropout=embed_dropout, 
            attention_dropout=attention_dropout, 
            pos_encoding=pos_encoding, 
            pos_len=pos_len
        )
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_timesteps: Optional[torch.Tensor]=None,
        tgt_timesteps: Optional[torch.Tensor]=None, 
        src_attention_mask: Optional[torch.Tensor]=None, 
        src_key_padding_mask: Optional[torch.Tensor]=None, 
        tgt_attention_mask: Optional[torch.Tensor]=None, 
        tgt_key_padding_mask: Optional[torch.Tensor]=None, 
        do_embedding: bool=True
    ):
        # Normally we don't need src_timesteps and tgt_timesteps for 
        # natural language processing tasks, but for RL tasks, we may cut 
        # trajectories into trunks and forward them trunk by trunk, and thus
        # the interface for passing in timesteps here is necessary
        enc_src = self.encoder(
            inputs=src, 
            timesteps=src_timesteps, 
            attention_mask=src_attention_mask, 
            key_padding_task=src_key_padding_mask, 
            do_embedding=do_embedding
        )
        output = self.decoder(
            tgt=tgt, 
            enc_src=enc_src, 
            timesteps=tgt_timesteps, 
            tgt_attention_mask=tgt_attention_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            src_key_padding_mask=src_key_padding_mask, 
            do_embedding=do_embedding
        )
        return output
        