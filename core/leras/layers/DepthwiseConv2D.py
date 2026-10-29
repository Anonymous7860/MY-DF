import numpy as np
import torch
import torch.nn.functional as F
from core.leras import nn

class DepthwiseConv2D(nn.LayerBase):
    """
    default kernel_initializer - CA
    use_wscale  bool enables equalized learning rate, if kernel_initializer is None, it will be forced to random_normal
    """
    def __init__(self, in_ch, kernel_size, strides=1, padding='SAME', depth_multiplier=1, dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
        if not isinstance(strides, int):
            raise ValueError ("strides must be an int type")
        if not isinstance(dilations, int):
            raise ValueError ("dilations must be an int type")
        kernel_size = int(kernel_size)

        if dtype is None:
            dtype = nn.floatx

        # Padding logic match Leras Conv2D
        if isinstance(padding, str):
            if padding == "SAME":
                padding = ( (kernel_size - 1) * dilations + 1 ) // 2
            elif padding == "VALID":
                padding = 0
            else:
                raise ValueError ("Wrong padding type. Should be VALID SAME or INT")
        else:
            padding = int(padding)
            
        self.in_ch = in_ch
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        # PyTorch Depthwise Conv: groups=in_ch.
        # Output channels = in_ch * depth_multiplier
        # Weight shape: (out_channels, in_channels/groups, k, k)
        #             = (in_ch * depth_multiplier, 1, k, k)
        
        self.out_ch = self.in_ch * self.depth_multiplier
        weight_shape = (self.out_ch, 1, self.kernel_size, self.kernel_size)
        
        # Initializers
        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else np.sqrt(2)
            fan_in = self.kernel_size*self.kernel_size*self.in_ch # ? Depthwise fan_in logic?
            # Actually fan_in is k*k*1 for depthwise usually?
            # But stick to Leras logic matching if possible.
            # Leras depthwise defaults to He Normal?
            he_std = gain / np.sqrt(fan_in)
            self.wscale = torch.tensor(he_std, dtype=self.dtype)
            init_fn = lambda t: torch.nn.init.normal_(t, 0.0, 1.0)
        else:
            self.wscale = None
            if self.kernel_initializer is None:
                # Kaiming initialization uses the layer's LeakyReLU slope.
                init_fn = lambda t: torch.nn.init.kaiming_normal_(t, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            else:
                 init_fn = lambda t: torch.nn.init.kaiming_normal_(t, a=0, mode='fan_in', nonlinearity='relu')

        self.weight = torch.nn.Parameter(torch.empty(weight_shape, dtype=self.dtype))
        init_fn(self.weight)
        
        if self.trainable:
             self.weight.requires_grad = True
        else:
             self.weight.requires_grad = False

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_ch, dtype=self.dtype))
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
        if nn.data_format == "NHWC":
            x = x.permute(0, 3, 1, 2)
            
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale.to(weight.device)

        x = F.conv2d(x, weight, bias=self.bias, stride=self.strides, padding=self.padding, 
                     dilation=self.dilations, groups=self.in_ch)
        
        if nn.data_format == "NHWC":
             x = x.permute(0, 2, 3, 1)
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} depth_multiplier:{self.depth_multiplier} "
        return r

nn.DepthwiseConv2D = DepthwiseConv2D
