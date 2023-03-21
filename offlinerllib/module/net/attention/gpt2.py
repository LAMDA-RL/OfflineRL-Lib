from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn


class GPTBlock(nn.Module):
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
        residual = residual + self.drop(attn_output) # this is because pytorch MHV don't do dropout after final projection
        residual = residual + self.ff(self.ln2(residual))
        return residual
        
        
class GPT2(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        num_heads: int, 
        causal: bool=False, 
        attention_dropout: Optional[float]=0.1, 
        residual_dropout: Optional[float]=0.1, 
        embed_dropout: Optional[float]=0.1, 
        position_dim: Optional[int]=None
    ) -> None:
        super().__init__()
        self.input_embed = nn.Linear(input_dim, embed_dim)
        position_dim = position_dim or seq_len
        self.pos_embed = nn.Embedding(position_dim, embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout) if embed_dropout else nn.Identity()
        self.out_ln = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            GPTBlock(
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
        inputs = self.embed_dropout(inputs)  # check: we escape the layer norm, while only corl does ln
        for i, block in enumerate(self.blocks):
            inputs = block(inputs, attention_mask=mask, key_padding_mask=key_padding_mask)
        inputs = self.out_ln(inputs)
        return inputs
    
    def configure_params(self):
        # https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215
        # However in this method we are not returning an Optimizer, but the parameter groups which
        # need / needn't weight decay, to support for more flexible downstream application
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
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
            
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        return [param_dict[pn] for pn in sorted(list(decay))], [param_dict[pn] for pn in sorted(list(no_decay))]

