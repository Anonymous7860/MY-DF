import numpy as np
import torch
import torch.nn.functional as F
from core.leras import nn

class Dense(nn.LayerBase):
    def __init__(self, in_ch, out_ch, use_bias=True, use_wscale=False, maxout_ch=0, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
        """
        use_wscale          enables weight scale (equalized learning rate)
        """
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.maxout_ch = maxout_ch
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        if dtype is None:
            dtype = nn.floatx

        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        # Determine actual output channels for weight matrix
        if self.maxout_ch > 1:
            actual_out_ch = self.out_ch * self.maxout_ch
        else:
            actual_out_ch = self.out_ch

        # PyTorch F.linear uses weight (out_ch, in_ch)
        weight_shape = (actual_out_ch, self.in_ch)
        
        # Initializers
        if self.use_wscale:
            gain = 1.0
            # fan_in is in_ch for dense
            fan_in = self.in_ch 
            he_std = gain / np.sqrt(fan_in)
            self.wscale = torch.tensor(he_std, dtype=self.dtype)

            init_fn = self.kernel_initializer or (
                lambda t: torch.nn.init.normal_(t, 0.0, 1.0)
            )
        else:
            self.wscale = None
            # Preserve the current LeakyReLU-aware default while honoring an
            # initializer explicitly supplied through the public layer API.
            init_fn = self.kernel_initializer or (
                lambda t: torch.nn.init.kaiming_uniform_(
                    t,
                    a=0.1,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                )
            )

        self.weight = torch.nn.Parameter(torch.empty(weight_shape, dtype=self.dtype))
        init_fn(self.weight)
        
        if self.trainable:
            self.weight.requires_grad = True
        else:
             self.weight.requires_grad = False

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_ch, dtype=self.dtype))
            if self.bias_initializer is not None:
                self.bias_initializer(self.bias.data)
            if self.trainable:
                 self.bias.requires_grad = True
            else:
                 self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

    def get_weights(self):
        weights = [self.weight]
        if self.use_bias:
            weights += [self.bias]
        return weights

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale.to(weight.device)
            
        # x: (N, in_ch)
        # F.linear(x, weight) -> x @ weight.T -> (N, out_ch)
        
        # Note: Leras maxout logic applied BEFORE bias if maxout_ch > 1
        # because reshape happens.
        
        # If maxout, intermediate output is (N, out_ch * maxout_ch)
        x = F.linear(x, weight) 
        
        if self.maxout_ch > 1:
            # Reshape to (N, out_ch, maxout_ch)
            # PyTorch: x is (N, out*maxout)
            x = x.view(-1, self.out_ch, self.maxout_ch)
            x, _ = torch.max(x, dim=-1)
            # Now x is (N, out_ch)

        if self.use_bias:
            # bias is (out_ch,)
            x = x + self.bias
            
        return x
nn.Dense = Dense
