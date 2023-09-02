import torch
from torch import nn

def get_pos_encoding(cls: str, embed_dim: int, seq_len: int, *args, **kwargs):
    cls2enc = {
        "embed": PosEmbedding, 
        "none": DummyEncoding, 
        "dummy": DummyEncoding, 
        "sinusoidal": SinusoidalEncoding, 
        "rope": RotaryPosEmbedding
    }
    if cls not in cls2enc:
        raise ValueError(f"Invalid positional encoding: {cls}, choices are {list(cls2enc.keys())}.")
    return cls2enc[cls](embed_dim, seq_len, *args, **kwargs)


class BasePosEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

        
class PosEmbedding(BasePosEncoding):
    """
    Learnable Positional Embedding. <Ref: Language Models are Unsupervised Multitask Learners>
    """
    def __init__(self, embed_dim: int, seq_len: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(seq_len, embed_dim)

    def forward(self, x, timestep=None):
        if timestep is None:
            return x + self.embedding(torch.arange(x.shape[1]).to(x.device)).repeat(x.shape[0], 1, 1)
        else:
            return x + self.embedding(timestep)
    

class DummyEncoding(BasePosEncoding):
    """
    Dummy Encoding, i.e. no positional encoding. 
    """
    def __init__(self, embed_dim: int, seq_len: int):
        super().__init__()
        
    def forward(self, x, timestep=None):
        return x

    
class SinusoidalEncoding(BasePosEncoding):
    """
    Sinusoidal Encoding. <Ref: https://arxiv.org/abs/1706.03762>
    """
    def __init__(self, embed_dim: int, seq_len: int, base: int=10000):
        super().__init__()
        # It is strange that I found that 
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.base = base
        
    def forward(self, x, timestep=None):
        B, L, E = x.shape
        if timestep is None:
            timestep = torch.arange(L).to(x.device).repeat(B, 1)
        timestep = timestep.float().unsqueeze(-1)
        inv_freq = 1.0/(self.base**(torch.arange(0, self.embed_dim, step=2)/self.embed_dim)).to(x.device)
        return x + torch.stack([
                torch.sin(timestep * inv_freq), 
                torch.cos(timestep * inv_freq)
            ], dim=-1).reshape(B, L, E).detach()
        

class RotaryPosEmbedding(BasePosEncoding):
    """
    Rotary Positional Embedding. <Ref: https://arxiv.org/abs/2104.09864>
    """
    def __init__(self, embed_dim: int, seq_len: int, base: int=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.base = base
        
    def forward(self, x, timestep=None):
        B, L, E = x.shape
        if timestep is None:
            timestep = torch.arange(L).to(x.device).repeat(B, 1)
        timestep = timestep.float().unsqueeze(-1)
        freq = (self.base**(torch.arange(0, self.embed_dim, step=2)/self.embed_dim)).to(x.device)
        sin = torch.sin(timestep / freq)
        cos = torch.cos(timestep / freq)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([
            x1*cos - x2*sin, x1*sin + x2*cos
        ], axis=-1).reshape(B, L, E)

