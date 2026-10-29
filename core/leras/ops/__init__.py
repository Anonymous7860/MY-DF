import torch
import torch.nn.functional as F

from core.leras import nn

# ===== Image Processing =====

_gaussian_blur_cache = {}


def gaussian_blur(x, radius=2.0):
    """
    Apply Gaussian blur using separable 1D convolutions (horizontal + vertical).

    Mathematically identical to full 2D Gaussian convolution since the Gaussian
    kernel is separable: G2D(x,y) = G1D(x) * G1D(y). Two 1D passes with a
    size-K kernel cost O(2*K) per pixel vs O(K^2) for 2D, giving ~12x speedup
    at typical training resolutions (e.g. 896px with K=57).

    Args:
        x: Input image tensor (NCHW or NHWC)
        radius: Blur radius (sigma)

    Returns:
        Blurred image tensor (same dtype as input)
    """
    if nn.data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)

    channels = x.shape[1]
    device = x.device
    input_dtype = x.dtype

    cache_key = (radius, channels, device)
    if cache_key not in _gaussian_blur_cache:
        kernel_size = max(3, int(2 * 2 * radius))
        if kernel_size % 2 == 0:
            kernel_size += 1

        mean = (kernel_size - 1) / 2.0
        x_coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        kernel_1d = torch.exp(-(x_coords - mean).pow(2) / (2 * radius**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Horizontal kernel: (C, 1, 1, K)
        kernel_h = kernel_1d.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
        # Vertical kernel: (C, 1, K, 1)
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)

        if len(_gaussian_blur_cache) > 32:
            _gaussian_blur_cache.clear()

        _gaussian_blur_cache[cache_key] = (kernel_h, kernel_v, kernel_size)

    kernel_h, kernel_v, kernel_size = _gaussian_blur_cache[cache_key]
    padding = kernel_size // 2

    kernel_h_typed = kernel_h.to(input_dtype)
    kernel_v_typed = kernel_v.to(input_dtype)

    # Horizontal pass
    x = F.pad(x, [padding, padding, 0, 0], mode="reflect")
    x = F.conv2d(x, kernel_h_typed, groups=channels)
    # Vertical pass
    x = F.pad(x, [0, 0, padding, padding], mode="reflect")
    x = F.conv2d(x, kernel_v_typed, groups=channels)

    if nn.data_format == "NHWC":
        x = x.permute(0, 2, 3, 1)

    return x


nn.gaussian_blur = gaussian_blur


def style_loss(target, style, blur_radius=0.0, gaussian_blur_radius=None, loss_weight=1.0, step_size=1):
    """
    Match per-channel mean and variance statistics between two image tensors.

    Returns one loss value per sample and keeps the math in FP32 for stable
    gradients under autocast.
    """
    if gaussian_blur_radius is not None:
        blur_radius = gaussian_blur_radius

    if blur_radius > 0.0:
        target = gaussian_blur(target, blur_radius)
        style = gaussian_blur(style, blur_radius)

    if nn.data_format == "NHWC":
        target = target.permute(0, 3, 1, 2)
        style = style.permute(0, 3, 1, 2)

    target = target.float()
    style = style.float()

    nc = target.shape[1]
    t_mean = target.mean(dim=[2, 3], keepdim=True)
    t_var = target.var(dim=[2, 3], keepdim=True, correction=0)
    t_std = torch.sqrt(t_var + 1e-5)

    s_mean = style.mean(dim=[2, 3], keepdim=True)
    s_var = style.var(dim=[2, 3], keepdim=True, correction=0)
    s_std = torch.sqrt(s_var + 1e-5)

    mean_loss = ((t_mean - s_mean) ** 2).sum(dim=[1, 2, 3])
    std_loss = ((t_std - s_std) ** 2).sum(dim=[1, 2, 3])
    return (mean_loss + std_loss) * (loss_weight / nc)


nn.style_loss = style_loss
