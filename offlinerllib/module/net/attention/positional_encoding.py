import torch
from torch import nn

def get_pos_encoding(cls: str, embed_dim: int, seq_len: int, *args, **kwargs):
    cls2enc = {
        "embed": PosEmbedding(embed_dim, seq_len, *args, **kwargs), 
        "none": DummyEncoding(embed_dim, seq_len, *args, **kwargs), 
        "dummy": DummyEncoding(embed_dim, seq_len, *args, **kwargs), 
        "sinusoidal": SinusoidalEncoding(embed_dim, seq_len, *args, **kwargs), 
        "rope": RotaryPosEmbedding(embed_dim, seq_len, *args, **kwargs)
    }
    if cls not in cls2enc:
        raise ValueError(f"Invalid positional encoding: {cls}, choices are {list(cls2enc.keys())}.")
    return cls2enc[cls]


class BasePosEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        pass
    
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
            return x + self.embedding(torch.arange(x.shape[1])).repeat(x.shape[0], 1).to(x.device)
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

        self.register_buffer("encoding", torch.zeros(seq_len, embed_dim))

        pos = torch.arange(seq_len).float().unsqueeze(dim=1)
        _2i = torch.arange(0, embed_dim, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (base ** (_2i / embed_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (base ** (_2i / embed_dim)))

    def forward(self, x, timestep=None):
        if timestep is None:
            return x + self.encoding[torch.arange(x.shape[1]).to(x.device)].repeat(x.shape[0], 1)
        else:
            return x + self.encoding[timestep].reshape(*x.shape)
        

class RotaryPosEmbedding(BasePosEncoding):
    """
    Rotary Positional Embedding. <Ref: https://arxiv.org/abs/2104.09864>
    """
    def __init__(self, embed_dim: int, seq_len: int, base: int=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        t = torch.arange(seq_len).float()
        freqs = torch.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())
        
    def forward(self, x, timestep=None):
        if timestep is None:
            timestep = torch.arange(x.shape[1]).to(x.device)
            sin = self.sin[timestep, :].repeat(x.shape[0], 1)
            cos = self.cos[timestep, :].repeat(x.shape[0], 1)
        else:
            sin, cos = self.sin[timestep, :], self.cos[timestep, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        ret = torch.stack([
            x1*cos - x2*sin, x1*sin + x2*cos
        ], axis=-1).reshape(*x.shape)
        return ret
        
