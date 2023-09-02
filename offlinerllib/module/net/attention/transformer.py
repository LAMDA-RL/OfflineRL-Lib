from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.base import BaseTransformer
from offlinerllib.module.net.attention.positional_encoding import get_pos_encoding


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        backbone_dim: Optional[int]=None, 
        pre_norm: bool=False, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None
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
        self.dropout1 = nn.Dropout(residual_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim), 
            nn.ReLU(), 
            nn.Dropout(residual_dropout), 
            nn.Linear(backbone_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(residual_dropout)
        self.pre_norm = pre_norm
        
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
        if self.pre_norm:
            residual = residual + self._sa_block(self.norm1(input), attention_mask, key_padding_mask)
            residual = residual + self._ff_block(self.norm2(residual))
        else:
            residual = self.norm1(residual + self._sa_block(input, attention_mask, key_padding_mask))
            residual = self.norm2(residual + self._ff_block(residual))
        return residual
        
    def _sa_block(self, input, attention_mask, key_padding_mask):
        input = self.attention(
            query=input, 
            key=input, 
            value=input, 
            need_weights=False, 
            attn_mask=attention_mask, 
            key_padding_mask=key_padding_mask
        )[0]
        return self.dropout1(input)
    
    def _ff_block(self, input):
        return self.dropout2(self.ff(input))
    

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        backbone_dim: Optional[int]=None, 
        pre_norm: bool=False, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None
    ) -> None:
        super().__init__()
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=attention_dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(residual_dropout)
        
        self.enc_dec_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=attention_dropout, 
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(residual_dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim), 
            nn.ReLU(), 
            nn.Dropout(residual_dropout), 
            nn.Linear(backbone_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(residual_dropout)
        self.pre_norm = pre_norm
        
    def forward(
        self, 
        tgt: torch.Tensor, 
        enc_src: Optional[torch.Tensor]=None, 
        tgt_attention_mask: Optional[torch.Tensor]=None, 
        tgt_key_padding_mask: Optional[torch.Tensor]=None, 
        src_attention_mask: Optional[torch.Tensor]=None, 
        src_key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        residual = tgt
        if self.pre_norm:
            residual = residual + self._sa_block(self.norm1(residual), tgt_attention_mask, tgt_key_padding_mask)
            if enc_src is not None:
                residual = residual + self._mha_block(self.norm2(residual), enc_src, src_attention_mask, src_key_padding_mask)
            residual = residual + self._ff_block(self.norm3(residual))
        else:
            residual = self.norm1(residual + self._sa_block(residual, tgt_attention_mask, tgt_key_padding_mask))
            if enc_src is not None:
                residual = self.norm2(residual + self._mha_block(residual, enc_src, src_attention_mask, src_key_padding_mask))
            residual = self.norm3(residual + self._ff_block(residual))
        return residual
    
    def _sa_block(self, input, attention_mask, key_padding_mask):
        input = self.self_attention(
            query=input, 
            key=input, 
            value=input, 
            need_weights=False, 
            attn_mask=attention_mask, 
            key_padding_mask=key_padding_mask
        )[0]
        return self.dropout1(input)
    
    def _mha_block(self, input, key_value, attention_mask, key_padding_mask):
        input = self.enc_dec_attention(
            query=input, 
            key=key_value, 
            value=key_value, 
            need_weights=False, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask
        )[0]
        return self.dropout2(input)
    
    def _ff_block(self, input):
        return self.dropout3(self.ff(input))
        

class TransformerEncoder(BaseTransformer):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        num_heads: int, 
        causal: bool=False, 
        out_ln: bool=True, 
        pre_norm: bool=False, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        embed_dropout: Optional[float]=None, 
        pos_encoding: str="sinusoid", 
        seq_len: Optional[int]=None, 
    ) -> None:
        super().__init__()
        self.input_embed = nn.Linear(input_dim, embed_dim)
        seq_len = seq_len or 1024
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, seq_len)
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                pre_norm=pre_norm, 
                attention_dropout=attention_dropout, 
                residual_dropout=residual_dropout
            ) for _ in range(num_layers)
        ])
        
        self.out_ln = nn.LayerNorm() if out_ln else nn.Identity()
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
            inputs = self.pos_encoding(inputs, timesteps)
        inputs = self.embed_dropout(inputs)
        for i, block in enumerate(self.blocks):
            inputs = block(inputs, attention_mask=mask, key_padding_mask=key_padding_mask)
        return self.out_ln(inputs)
    

class TransformerDecoder(BaseTransformer):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        num_heads: int, 
        causal: bool=True, 
        out_ln: bool=True, 
        pre_norm: bool=False, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        embed_dropout: Optional[float]=None, 
        pos_encoding: str="sinusoid", 
        seq_len: Optional[int]=None, 
    ) -> None:
        super().__init__()
        self.input_embed = nn.Linear(input_dim, embed_dim)
        seq_len = seq_len or 1024
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, seq_len)
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                pre_norm=pre_norm, 
                attention_dropout=attention_dropout, 
                residual_dropout=residual_dropout
            ) for _ in range(num_layers)
        ])
        
        self.out_ln = nn.LayerNorm() if out_ln else nn.Identity()
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
        return self.out_ln(output)
        

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
        
        