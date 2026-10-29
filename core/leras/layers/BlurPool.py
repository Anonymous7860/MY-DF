import torch
import torch.nn.functional as F
from torch import nn as torch_nn
import numpy as np
from core.leras import nn

class BlurPool(torch_nn.Module):
    """
    EXACT DeepFaceLab / TensorFlow BlurPool behavior.
    NCHW only. Pixel-faithful. XSeg-safe.
    """

    def __init__(self, filt_size=3, stride=2):
        super().__init__()

        self.filt_size = filt_size
        self.stride = stride

        # --- padding calculation (EXACT TF logic) ---
        pad_l = int((filt_size - 1) / 2)
        pad_r = int(np.ceil((filt_size - 1) / 2))

        # PyTorch pad format: (left, right, top, bottom)
        self.pad = (pad_l, pad_r, pad_l, pad_r)

        # --- Pascal blur kernel ---
        if filt_size == 1:
            a = np.array([1.])
        elif filt_size == 2:
            a = np.array([1., 1.])
        elif filt_size == 3:
            a = np.array([1., 2., 1.])
        elif filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError("Unsupported filt_size")

        a = a[:, None] * a[None, :]
        a = a / np.sum(a)

        # Store as (1, 1, kH, kW) — will be expanded per-channel
        self.register_buffer(
            "kernel",
            torch.tensor(a, dtype=torch.float32)[None, None, :, :]
        )

    def forward(self, x):
        # XSeg MUST be NCHW
        assert x.ndim == 4, "BlurPool expects NCHW input"

        c = int(x.shape[1])

        # --- CONSTANT ZERO padding (matches tf.pad default) ---
        if self.pad != (0, 0, 0, 0):
            x = F.pad(x, self.pad, mode="constant", value=0)

        # --- depthwise kernel ---
        # TF: tile (kH, kW, C, 1)
        # PyTorch: (C, 1, kH, kW)
        kernel_name = f"kernel_c{c}"
        kernel = getattr(self, kernel_name, None)
        if kernel is None:
            kernel = self.kernel.repeat(c, 1, 1, 1)
            self.register_buffer(kernel_name, kernel, persistent=False)
        kernel = kernel.to(device=x.device, dtype=x.dtype)

        # --- depthwise convolution ---
        y = F.conv2d(
            x,
            kernel,
            stride=(self.stride, self.stride),
            padding=0,       # VALID
            groups=c
        )

        return y

# Alias for compatibility
nn.BlurPool = BlurPool
