import math
import torch
import torch.nn.functional as F
from core.leras import nn


class Conv2DTranspose(nn.LayerBase):
    """
    PyTorch implementation that EXACTLY mimics TensorFlow Conv2DTranspose
    behavior used in DeepFaceLab XSeg.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        strides=2,
        padding="SAME",
        use_bias=True,
        use_wscale=False,
        kernel_initializer=None,
        bias_initializer=None,
        trainable=True,
        dtype=None,
        **kwargs
    ):
        if not isinstance(strides, int):
            raise ValueError("strides must be int")

        if dtype is None:
            dtype = nn.floatx

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = int(kernel_size)
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        self.dtype = dtype

        super().__init__(**kwargs)

    # ------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------
    def build_weights(self):
        # TF weight layout: (kH, kW, out_ch, in_ch)
        # PyTorch expects: (in_ch, out_ch, kH, kW)
        weight_shape = (self.in_ch, self.out_ch, self.kernel_size, self.kernel_size)

        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else math.sqrt(2.0)
            fan_in = self.kernel_size * self.kernel_size * self.in_ch
            he_std = gain / math.sqrt(fan_in)
            self.wscale = torch.tensor(he_std, dtype=self.dtype)
            init = self.kernel_initializer or (
                lambda t: torch.nn.init.normal_(t, 0.0, 1.0)
            )
        else:
            self.wscale = None
            # Preserve the current LeakyReLU-aware default while honoring an
            # initializer explicitly supplied through the public layer API.
            init = self.kernel_initializer or (
                lambda t: torch.nn.init.kaiming_normal_(
                    t,
                    a=0.1,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                )
            )

        self.weight = torch.nn.Parameter(torch.empty(weight_shape, dtype=self.dtype))
        init(self.weight)
        self.weight.requires_grad = self.trainable

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_ch, dtype=self.dtype))
            if self.bias_initializer is not None:
                self.bias_initializer(self.bias.data)
            self.bias.requires_grad = self.trainable
        else:
            self.register_parameter("bias", None)

    # ------------------------------------------------------------
    # Output size logic (matches TF deconv_length)
    # ------------------------------------------------------------
    @staticmethod
    def _deconv_length(dim, stride, kernel, padding):
        if padding == "VALID":
            return dim * stride + max(kernel - stride, 0)
        elif padding == "FULL":
            return dim * stride - (stride + kernel - 2)
        elif padding == "SAME":
            return dim * stride
        else:
            raise ValueError("Invalid padding")

    def _conv_transpose_adjustment(self):
        """Return front crop, back crop, and output growth for TF padding."""
        kernel = self.kernel_size
        stride = self.strides

        if self.padding == "SAME":
            crop_total = max(kernel - stride, 0)
            crop_before = crop_total // 2
            crop_after = crop_total - crop_before
            grow = max(stride - kernel, 0)
        elif self.padding == "VALID":
            crop_before = 0
            crop_after = 0
            grow = max(stride - kernel, 0)
        elif self.padding == "FULL":
            crop_before = kernel - 1
            crop_after = kernel - 1
            grow = 0
        else:
            raise ValueError(f"Invalid padding: {self.padding}")

        return crop_before, crop_after, grow

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x):
        # XSeg MUST be NCHW
        assert nn.data_format != "NHWC", "XSeg Conv2DTranspose must be NCHW"

        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale.to(weight.device)

        # Note: Both TensorFlow and PyTorch conv_transpose2d use correlation (not convolution),
        # so NO kernel flip is needed - removing the incorrect flip that was causing spatial misalignment

        # TensorFlow SAME may require asymmetric cropping, while kernels
        # smaller than the stride require PyTorch output growth.
        crop_before, crop_after, grow = self._conv_transpose_adjustment()

        # Generate the full transpose-convolution result before cropping.
        x = F.conv_transpose2d(
            x,
            weight,
            bias=None,
            stride=self.strides,
            padding=0,
            output_padding=grow,
        )

        # Crop both spatial axes using TensorFlow's padding split.
        if crop_before or crop_after:
            end_h = x.shape[2] - crop_after if crop_after else x.shape[2]
            end_w = x.shape[3] - crop_after if crop_after else x.shape[3]
            x = x[:, :, crop_before:end_h, crop_before:end_w]

        if self.use_bias:
            x = x + self.bias.view(1, -1, 1, 1)

        return x

    def __str__(self):
        return f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch}"


nn.Conv2DTranspose = Conv2DTranspose
