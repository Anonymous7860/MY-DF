import torch
from core.leras import nn

class DenseNorm(nn.LayerBase):
    def __init__(self, dense=False, eps=1e-06, dtype=None, **kwargs):
        self.dense = dense        
        if dtype is None:
            dtype = nn.floatx
        # In PyTorch, scalars can be plain numbers or 0-dim tensor.
        self.eps = eps 
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        # No weights for this layer?
        pass

    def forward(self, x):
        # x * rsqrt(mean(x^2) + eps)
        mean_sq = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_sq + self.eps)
        
nn.DenseNorm = DenseNorm
