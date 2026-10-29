import torch
from core.leras import nn

class TLU(nn.LayerBase):
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
        self.tau = torch.nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype))

    def get_weights(self):
        return [self.tau]

    def forward(self, x):
        # tau needs to be broadcastable.
        # x: (N, C, H, W) or (N, C)?
        # Leras: reshape tau to (1,1,1,C) or (1,C,1,1).
        
        tau = self.tau
        if nn.data_format == "NHWC":
            tau = tau.view(1, 1, 1, self.in_ch)
        else:
            # Check dimensions of x.
            if x.dim() == 4:
                tau = tau.view(1, self.in_ch, 1, 1)
            elif x.dim() == 2:
                 tau = tau.view(1, self.in_ch)
        
        return torch.maximum(x, tau)

nn.TLU = TLU