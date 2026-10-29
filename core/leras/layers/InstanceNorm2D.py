import torch
import torch.nn.functional as F
from core.leras import nn

class InstanceNorm2D(nn.LayerBase):
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        # We can wrap nn.InstanceNorm2d, but Leras splits weight/bias initialization?
        # Leras uses Glorot Uniform for weight, Zeros for bias.
        # nn.InstanceNorm2d defaults: weight=1, bias=0.
        
        # To strictly match Leras init, we manipulate parameters.
        # However, for InstanceNorm, affine weights usually start at 1.
        # Glorot Uniform for scale parameter in IN?? That seems odd (it's usually for weights in Linear/Conv).
        # Leras: tf.get_variable("weight", ..., initializer=glorot_uniform)
        
        self.weight = torch.nn.Parameter(torch.empty(self.in_ch, dtype=self.dtype))
        # Unit scale makes normalization identity-scaled at initialization.
        torch.nn.init.ones_(self.weight)
        # Let's assume user wanted some variance.
        
        self.bias = torch.nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype))
        
        # We don't use nn.InstanceNorm2d module because we want full control over forward to match Leras exact arithmetic if needed?
        # But F.instance_norm is available.
    
    def get_weights(self):
        return [self.weight, self.bias]

    def forward(self, x):
        if nn.data_format == "NHWC":
            x = x.permute(0, 3, 1, 2)
            
        # F.instance_norm(input, running_mean=None, running_var=None, weight=self.weight, bias=self.bias, use_input_stats=True, ...)
        # Wait, F.instance_norm interface:
        # (input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05)
        
        x = F.instance_norm(x, weight=self.weight, bias=self.bias, eps=1e-5)
        
        if nn.data_format == "NHWC":
            x = x.permute(0, 2, 3, 1)

        return x

nn.InstanceNorm2D = InstanceNorm2D
