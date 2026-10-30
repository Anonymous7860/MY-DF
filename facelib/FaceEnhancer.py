from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn


def _resize2d_bilinear(x, scale=2):
    """Match TensorFlow 1.x bilinear resizing used to train the supplied weights."""
    height, width = x.shape[-2:]
    output_height = height * scale
    output_width = width * scale
    dtype = x.dtype
    device = x.device

    source_y = torch.arange(output_height, dtype=torch.float32, device=device) / scale
    y0 = torch.floor(source_y).long().clamp(0, height - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    fraction_y = (source_y - y0.float()).to(dtype).view(1, 1, output_height, 1)

    source_x = torch.arange(output_width, dtype=torch.float32, device=device) / scale
    x0 = torch.floor(source_x).long().clamp(0, width - 1)
    x1 = (x0 + 1).clamp(max=width - 1)
    fraction_x = (source_x - x0.float()).to(dtype).view(1, 1, 1, output_width)

    rows = x.index_select(2, y0) * (1 - fraction_y) + x.index_select(2, y1) * fraction_y
    return rows.index_select(3, x0) * (1 - fraction_x) + rows.index_select(3, x1) * fraction_x


class FaceEnhancer(object):
    """Run the pretrained x4 face-enhancement network on BGR images."""

    def __init__(self, place_model_on_cpu=False, run_on_cpu=False):
        nn.initialize()

        class FaceEnhancerModel(nn.ModelBase):
            def __init__(self, name="FaceEnhancer"):
                super().__init__(name=name)

            def on_build(self):
                self.conv1 = nn.Conv2D(3, 64, kernel_size=3, strides=1, padding="SAME")

                self.dense1 = nn.Dense(1, 64, use_bias=False)
                self.dense2 = nn.Dense(1, 64, use_bias=False)

                self.e0_conv0 = nn.Conv2D(64, 64, kernel_size=3, strides=1, padding="SAME")
                self.e0_conv1 = nn.Conv2D(64, 64, kernel_size=3, strides=1, padding="SAME")

                self.e1_conv0 = nn.Conv2D(64, 112, kernel_size=3, strides=1, padding="SAME")
                self.e1_conv1 = nn.Conv2D(112, 112, kernel_size=3, strides=1, padding="SAME")

                self.e2_conv0 = nn.Conv2D(112, 192, kernel_size=3, strides=1, padding="SAME")
                self.e2_conv1 = nn.Conv2D(192, 192, kernel_size=3, strides=1, padding="SAME")

                self.e3_conv0 = nn.Conv2D(192, 336, kernel_size=3, strides=1, padding="SAME")
                self.e3_conv1 = nn.Conv2D(336, 336, kernel_size=3, strides=1, padding="SAME")

                self.e4_conv0 = nn.Conv2D(336, 512, kernel_size=3, strides=1, padding="SAME")
                self.e4_conv1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding="SAME")

                self.center_conv0 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding="SAME")
                self.center_conv1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding="SAME")
                self.center_conv2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding="SAME")
                self.center_conv3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding="SAME")

                self.d4_conv0 = nn.Conv2D(1024, 512, kernel_size=3, strides=1, padding="SAME")
                self.d4_conv1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding="SAME")

                self.d3_conv0 = nn.Conv2D(848, 512, kernel_size=3, strides=1, padding="SAME")
                self.d3_conv1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding="SAME")

                self.d2_conv0 = nn.Conv2D(704, 288, kernel_size=3, strides=1, padding="SAME")
                self.d2_conv1 = nn.Conv2D(288, 288, kernel_size=3, strides=1, padding="SAME")

                self.d1_conv0 = nn.Conv2D(400, 160, kernel_size=3, strides=1, padding="SAME")
                self.d1_conv1 = nn.Conv2D(160, 160, kernel_size=3, strides=1, padding="SAME")

                self.d0_conv0 = nn.Conv2D(224, 96, kernel_size=3, strides=1, padding="SAME")
                self.d0_conv1 = nn.Conv2D(96, 96, kernel_size=3, strides=1, padding="SAME")

                self.out1x_conv0 = nn.Conv2D(96, 48, kernel_size=3, strides=1, padding="SAME")
                self.out1x_conv1 = nn.Conv2D(48, 3, kernel_size=3, strides=1, padding="SAME")

                self.dec2x_conv0 = nn.Conv2D(96, 96, kernel_size=3, strides=1, padding="SAME")
                self.dec2x_conv1 = nn.Conv2D(96, 96, kernel_size=3, strides=1, padding="SAME")

                self.out2x_conv0 = nn.Conv2D(96, 48, kernel_size=3, strides=1, padding="SAME")
                self.out2x_conv1 = nn.Conv2D(48, 3, kernel_size=3, strides=1, padding="SAME")

                self.dec4x_conv0 = nn.Conv2D(96, 72, kernel_size=3, strides=1, padding="SAME")
                self.dec4x_conv1 = nn.Conv2D(72, 72, kernel_size=3, strides=1, padding="SAME")

                self.out4x_conv0 = nn.Conv2D(72, 36, kernel_size=3, strides=1, padding="SAME")
                self.out4x_conv1 = nn.Conv2D(36, 3, kernel_size=3, strides=1, padding="SAME")

            def forward(self, bgr, param, param1):
                x = self.conv1(bgr)
                a = self.dense1(param).reshape(-1, 64, 1, 1)
                b = self.dense2(param1).reshape(-1, 64, 1, 1)
                x = F.leaky_relu(x + a + b, 0.1)

                x = F.leaky_relu(self.e0_conv0(x), 0.1)
                x = e0 = F.leaky_relu(self.e0_conv1(x), 0.1)

                x = F.avg_pool2d(x, 2, 2)
                x = F.leaky_relu(self.e1_conv0(x), 0.1)
                x = e1 = F.leaky_relu(self.e1_conv1(x), 0.1)

                x = F.avg_pool2d(x, 2, 2)
                x = F.leaky_relu(self.e2_conv0(x), 0.1)
                x = e2 = F.leaky_relu(self.e2_conv1(x), 0.1)

                x = F.avg_pool2d(x, 2, 2)
                x = F.leaky_relu(self.e3_conv0(x), 0.1)
                x = e3 = F.leaky_relu(self.e3_conv1(x), 0.1)

                x = F.avg_pool2d(x, 2, 2)
                x = F.leaky_relu(self.e4_conv0(x), 0.1)
                x = e4 = F.leaky_relu(self.e4_conv1(x), 0.1)

                x = F.avg_pool2d(x, 2, 2)
                x = F.leaky_relu(self.center_conv0(x), 0.1)
                x = F.leaky_relu(self.center_conv1(x), 0.1)
                x = F.leaky_relu(self.center_conv2(x), 0.1)
                x = F.leaky_relu(self.center_conv3(x), 0.1)

                x = torch.cat((_resize2d_bilinear(x), e4), dim=1)
                x = F.leaky_relu(self.d4_conv0(x), 0.1)
                x = F.leaky_relu(self.d4_conv1(x), 0.1)

                x = torch.cat((_resize2d_bilinear(x), e3), dim=1)
                x = F.leaky_relu(self.d3_conv0(x), 0.1)
                x = F.leaky_relu(self.d3_conv1(x), 0.1)

                x = torch.cat((_resize2d_bilinear(x), e2), dim=1)
                x = F.leaky_relu(self.d2_conv0(x), 0.1)
                x = F.leaky_relu(self.d2_conv1(x), 0.1)

                x = torch.cat((_resize2d_bilinear(x), e1), dim=1)
                x = F.leaky_relu(self.d1_conv0(x), 0.1)
                x = F.leaky_relu(self.d1_conv1(x), 0.1)

                x = torch.cat((_resize2d_bilinear(x), e0), dim=1)
                x = F.leaky_relu(self.d0_conv0(x), 0.1)
                d0 = F.leaky_relu(self.d0_conv1(x), 0.1)

                x = F.leaky_relu(self.out1x_conv0(d0), 0.1)
                out1x = bgr + torch.tanh(self.out1x_conv1(x))

                x = F.leaky_relu(self.dec2x_conv0(d0), 0.1)
                x = F.leaky_relu(self.dec2x_conv1(x), 0.1)
                d2x = _resize2d_bilinear(x)

                x = F.leaky_relu(self.out2x_conv0(d2x), 0.1)
                out2x = _resize2d_bilinear(out1x) + torch.tanh(self.out2x_conv1(x))

                x = F.leaky_relu(self.dec4x_conv0(d2x), 0.1)
                x = F.leaky_relu(self.dec4x_conv1(x), 0.1)
                d4x = _resize2d_bilinear(x)

                x = F.leaky_relu(self.out4x_conv0(d4x), 0.1)
                return _resize2d_bilinear(out2x) + torch.tanh(self.out4x_conv1(x))

        model_path = Path(__file__).parent / "FaceEnhancer.npy"
        if not model_path.exists():
            raise FileNotFoundError(f"Unable to load {model_path.name}")

        # A PyTorch module's parameters and operations must share a device.
        # Keep the legacy argument for API compatibility; execution placement
        # is therefore controlled by run_on_cpu.
        self.place_model_on_cpu = bool(place_model_on_cpu)
        self.device = torch.device("cpu") if run_on_cpu else nn.device

        self.model = FaceEnhancerModel()
        self.model.build()
        self.model.to(self.device)
        if not self.model.load_weights(model_path):
            raise RuntimeError(f"Unable to load weights from {model_path}")
        self.model.eval()

    def _net_run(self, patch, param, param1):
        """Convert an NHWC NumPy patch to NCHW for inference and back."""

        def to_tensor(array, shape=None):
            tensor = torch.as_tensor(np.ascontiguousarray(array), device=self.device, dtype=nn.floatx)
            return tensor if shape is None else tensor.reshape(shape)

        patch_tensor = to_tensor(patch).permute(0, 3, 1, 2)
        with torch.inference_mode():
            output = self.model(
                patch_tensor,
                to_tensor(param, (1, 1)),
                to_tensor(param1, (1, 1)),
            )
        return output.permute(0, 2, 3, 1).cpu().numpy()

    def enhance(self, inp_img, is_tanh=False, preserve_size=True):
        """Enhance an HWC BGR float image, optionally preserving its size."""
        if inp_img.ndim != 3 or inp_img.shape[2] != 3:
            raise ValueError(f"Expected an HWC BGR image with 3 channels, got {inp_img.shape}")

        inp_img = np.asarray(inp_img, dtype=np.float32)
        if not is_tanh:
            inp_img = np.clip(inp_img * 2.0 - 1.0, -1.0, 1.0)

        param = np.array([0.2], dtype=np.float32)
        param1 = np.array([1.0], dtype=np.float32)
        up_res = 4
        patch_size = 192
        patch_stride = patch_size // 2

        input_height, input_width, channels = inp_img.shape
        height, width = input_height, input_width

        top_padding = max(0, (patch_size - height) // 2)
        bottom_padding = max(0, patch_size - height - top_padding)
        left_padding = max(0, (patch_size - width) // 2)
        right_padding = max(0, patch_size - width - left_padding)

        if top_padding or bottom_padding or left_padding or right_padding:
            inp_img = np.pad(
                inp_img,
                ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)),
                mode="constant",
            )
            height, width, channels = inp_img.shape

        max_x = width - patch_size
        max_y = height - patch_size
        output = np.zeros((height * up_res, width * up_res, channels), dtype=np.float32)
        divisor = np.zeros((height * up_res, width * up_res, 1), dtype=np.float32)

        ramp = np.concatenate(
            (
                np.linspace(0.0, 1.0, patch_stride * up_res),
                np.linspace(1.0, 0.0, patch_stride * up_res),
            )
        )
        half_output_patch = patch_stride * up_res

        y = 0
        while True:
            y_weights = ramp.copy()
            if y == 0:
                y_weights[:half_output_patch] = 1.0
            if y == max_y:
                y_weights[half_output_patch:] = 1.0

            x = 0
            while True:
                x_weights = ramp.copy()
                if x == 0:
                    x_weights[:half_output_patch] = 1.0
                if x == max_x:
                    x_weights[half_output_patch:] = 1.0
                patch_mask = (y_weights[:, None] * x_weights[None, :])[..., None]

                patch = inp_img[y:y + patch_size, x:x + patch_size, :]
                enhanced = self._net_run(patch[None, ...], param, param1)[0]
                output[
                    y * up_res:(y + patch_size) * up_res,
                    x * up_res:(x + patch_size) * up_res,
                    :,
                ] += enhanced * patch_mask
                divisor[
                    y * up_res:(y + patch_size) * up_res,
                    x * up_res:(x + patch_size) * up_res,
                    :,
                ] += patch_mask

                if x == max_x:
                    break
                x = min(x + patch_stride, max_x)

            if y == max_y:
                break
            y = min(y + patch_stride, max_y)

        divisor[divisor == 0] = 1.0
        output /= divisor

        if top_padding or bottom_padding or left_padding or right_padding:
            output = output[
                top_padding * up_res:(height - bottom_padding) * up_res,
                left_padding * up_res:(width - right_padding) * up_res,
                :,
            ]

        if preserve_size:
            output = cv2.resize(
                output,
                (input_width, input_height),
                interpolation=cv2.INTER_LANCZOS4,
            )

        if not is_tanh:
            output = np.clip(output / 2.0 + 0.5, 0.0, 1.0)

        return output
