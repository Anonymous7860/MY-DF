import torch
import torch.nn.functional as F
from core.leras import nn

class AdaIN(nn.LayerBase):
    """
    Adaptive Instance Normalization
    """
    def __init__(self, in_ch, mlp_ch, kernel_initializer=None, dtype=None, **kwargs):
        self.in_ch = in_ch
        self.mlp_ch = mlp_ch
        self.kernel_initializer = kernel_initializer

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        # Maps mlp_ch -> in_ch
        # PyTorch Linear weight: (out_features, in_features) -> (in_ch, mlp_ch)
        
        weight_shape = (self.in_ch, self.mlp_ch)
        
        # Initializers
        if self.kernel_initializer is None:
            init_fn = torch.nn.init.kaiming_normal_
        else:
            # Fallback
            init_fn = torch.nn.init.kaiming_normal_

        self.weight1 = torch.nn.Parameter(torch.empty(weight_shape, dtype=self.dtype))
        init_fn(self.weight1)
        self.bias1 = torch.nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype))
        
        self.weight2 = torch.nn.Parameter(torch.empty(weight_shape, dtype=self.dtype))
        init_fn(self.weight2)
        self.bias2 = torch.nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype))

    def get_weights(self):
        return [self.weight1, self.bias1, self.weight2, self.bias2]

    def forward(self, inputs):
        x, mlp = inputs
        # mlp: (N, mlp_ch)
        # x: (N, in_ch, H, W) (assuming NCHW)
        
        if nn.data_format == "NHWC":
            x = x.permute(0, 3, 1, 2)

        # Style projection
        gamma = F.linear(mlp, self.weight1, self.bias1) # (N, in_ch)
        beta  = F.linear(mlp, self.weight2, self.bias2)
        
        # Reshape for broadcast: (N, in_ch, 1, 1)
        gamma = gamma.view(-1, self.in_ch, 1, 1) + 1.0 # AdaIN usually adds 1 to gamma? 
        # Wait, Leras code: gamma = matmul(...); gamma = add(gamma, bias)
        # Leras does NOT add 1 explicitly in the math shown?
        # "x *= tf.reshape(gamma, shape)"
        # Unless 'bias1' initialization handles it (zeros).
        # Standard AdaIN: norm(x) * (gamma) + beta.
        # If gamma starts at 0, output is 0 + beta? That kills structure at init.
        # Usually gamma init to 1, or bias init to 1.
        # Leras init: He Normal (random). Bias zeros.
        # So gamma is random around 0.
        # This implies it might be unstable at start, or He init provides variance.
        # I will match Leras: just raw linear output.
        
        beta = beta.view(-1, self.in_ch, 1, 1)

        # Instance Norm logic (without affine)
        # x_mean = mean(spatial)
        # x = (x - mean) / std
        
        # F.instance_norm with NO weight/bias does exactly this.
        x = F.instance_norm(x, eps=1e-5)
        
        x = x * gamma + beta
        
        if nn.data_format == "NHWC":
            x = x.permute(0, 2, 3, 1)

        return x

nn.AdaIN = AdaIN