import torch
from core.leras import nn

class ScaleAdd(nn.LayerBase):
    def __init__(self, ch, dtype=None, **kwargs):
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        self.ch = ch

        super().__init__(**kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(torch.zeros(self.ch, dtype=self.dtype))

    def get_weights(self):
        return [self.weight]

    def forward(self, inputs):
        # x0 + x1 * weight
        x0, x1 = inputs
        
        # Broadcast weight
        # Assuming inputs are (N, C) or (N, C, H, W)
        if hasattr(x1, 'dim') and x1.dim() == 4:
            if nn.data_format == "NHWC":
                 w = self.weight.view(1, 1, 1, self.ch)
            else:
                 w = self.weight.view(1, self.ch, 1, 1)
        else:
            if nn.data_format == "NHWC":
                 w = self.weight.view(1, self.ch)
            else:
                 w = self.weight.view(1, self.ch)

        return x0 + x1 * w

nn.ScaleAdd = ScaleAdd