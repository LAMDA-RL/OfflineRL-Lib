import torch
from torch import nn

from offlinerllib.module.net.attention.base import PositionalEncoding


class SinusoidEncoding(PositionalEncoding):
    """
    Sinusoid encoding.
    """
    def __init__(self, embed_dim, pos_len):
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(pos_len, embed_dim)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, pos_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, embed_dim, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_dim)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        return torch.gather(self.encoding, 0, x).detach()


class PositionalEmbedding(PositionalEncoding):
    """
    Direct embedding.
    """
    def __init__(self, embed_dim, pos_len):
        super().__init__(num_embeddings=pos_len, embedding_dim=embed_dim)


class ZeroEncoding(PositionalEncoding):
    """
    Zero encoding, i.e. no timestep encoding.
    """
    def __init__(self, embed_dim, pos_len):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, x):
        B, L = x.shape
        return torch.zeros([B, L, self.embed_dim]).to(x.device).detach()
    