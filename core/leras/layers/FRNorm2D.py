import torch
from core.leras import nn

class FRNorm2D(nn.LayerBase):
    """
    Filter Response Normalization Layer: Eliminating Batch Dependence in theTraining of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        # Learnable parameters
        self.weight = torch.nn.Parameter(torch.ones(self.in_ch, dtype=self.dtype))
        self.bias = torch.nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype))
        self.eps = torch.nn.Parameter(torch.tensor([1e-6], dtype=self.dtype))

    def get_weights(self):
        return [self.weight, self.bias, self.eps]

    def forward(self, x):
        if nn.data_format == "NHWC":
            x = x.permute(0, 3, 1, 2)
        
        # FRNorm stats (spatial)
        # nu2 = mean(x^2, dim=(2,3))
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        # x = x / sqrt(nu2 + abs(eps))
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        
        # Apply affine: weight/bias broadcast over spatial dims
        # weight/bias (C) -> (1, C, 1, 1)
        w = self.weight.view(1, self.in_ch, 1, 1)
        b = self.bias.view(1, self.in_ch, 1, 1)
        
        x = x * w + b
        
        if nn.data_format == "NHWC":
             x = x.permute(0, 2, 3, 1)
             
        return x

nn.FRNorm2D = FRNorm2D