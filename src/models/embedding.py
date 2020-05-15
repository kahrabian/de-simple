import torch as T
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, r_dim):
        super(PositionalEmbedding, self).__init__()

        frq = 1 / (10000 ** (T.arange(0.0, r_dim, 2.0) / r_dim))
        self.register_buffer('frq', frq)

    def forward(self, r):
        r_sin = T.ger(r, self.frq)
        return T.cat([r_sin.sin(), r_sin.cos()], dim=-1)
