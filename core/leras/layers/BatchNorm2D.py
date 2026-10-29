import torch
import torch.nn.functional as F
from core.leras import nn

class BatchNorm2D(nn.LayerBase):
    """
    currently not for training - implies inference mode or static stats.
    """
    def __init__(self, dim, eps=1e-05, momentum=0.1, dtype=None, **kwargs):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        # Affine parameters
        self.weight = torch.nn.Parameter(torch.ones(self.dim, dtype=self.dtype))
        self.bias = torch.nn.Parameter(torch.zeros(self.dim, dtype=self.dtype))
        
        # Running stats - registered as buffers (not parameters, but state)
        self.register_buffer('running_mean', torch.zeros(self.dim, dtype=self.dtype))
        self.register_buffer('running_var', torch.zeros(self.dim, dtype=self.dtype))

    def get_weights(self):
        # Leras returned all 4.
        return [self.weight, self.bias, self.running_mean, self.running_var]

    def forward(self, x):
        if nn.data_format == "NHWC":
             x = x.permute(0, 3, 1, 2)
             
        # F.batch_norm with training=True updates running stats.
        # F.batch_norm with training=False uses running stats.
        # The module's training state selects running-stat updates or inference statistics.
        x = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                         training=self.training, momentum=self.momentum, eps=self.eps)
                         
        if nn.data_format == "NHWC":
             x = x.permute(0, 2, 3, 1)
        return x

nn.BatchNorm2D = BatchNorm2D
