import torch
from pytorch_msssim import MS_SSIM, SSIM

MIN_MSSSIM_RESOLUTION = 176


def _ensure_odd(n: int) -> int:
    n = max(1, n)
    return n if n % 2 == 1 else n + 1


class SSIMCache:
    _ssim_instances: dict[tuple[int, int], SSIM] = {}
    _msssim_instances: dict[int, MS_SSIM] = {}

    @classmethod
    def get_ssim(cls, channel: int = 3, win_size: int = 11) -> SSIM:
        key = (channel, win_size)
        if key not in cls._ssim_instances:
            cls._ssim_instances[key] = SSIM(
                data_range=1.0,
                size_average=False,
                channel=channel,
                win_size=win_size,
            )
        return cls._ssim_instances[key]

    @classmethod
    def get_msssim(cls, channel: int = 3) -> MS_SSIM:
        if channel not in cls._msssim_instances:
            cls._msssim_instances[channel] = MS_SSIM(
                data_range=1.0,
                size_average=False,
                channel=channel,
            )
        return cls._msssim_instances[channel]

    @classmethod
    def clear(cls):
        cls._ssim_instances.clear()
        cls._msssim_instances.clear()


def ssim_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    channel: int = None,
    resolution: int = None,
    return_similarity: bool = False,
) -> torch.Tensor:
    """
    Compute SSIM loss matching legacy DFL dual-scale DSSIM formula.

    For resolution >= 256: 5*dssim(win=res/11.6) + 5*dssim(win=res/23.2)
    For resolution < 256:  10*dssim(win=res/11.6)
    dssim = (1 - SSIM) / 2

    Forces FP32 computation for numerical stability under BF16 autocast.

    Returns:
        Loss tensor [B] (always FP32)
    """
    if channel is None:
        channel = target.shape[1]
    if resolution is None:
        resolution = target.shape[2]

    device_type = "cuda" if target.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        target_f = target.float()
        pred_f = pred.float()

        win1 = _ensure_odd(int(resolution / 11.6))

        if resolution >= 256:
            win2 = _ensure_odd(int(resolution / 23.2))
            ssim1 = SSIMCache.get_ssim(channel, win1)(target_f, pred_f)
            ssim2 = SSIMCache.get_ssim(channel, win2)(target_f, pred_f)
            loss = 5.0 * (1.0 - ssim1) / 2.0 + 5.0 * (1.0 - ssim2) / 2.0
            similarity = ssim1
        else:
            ssim_val = SSIMCache.get_ssim(channel, win1)(target_f, pred_f)
            loss = 10.0 * (1.0 - ssim_val) / 2.0
            similarity = ssim_val

    if return_similarity:
        return loss, similarity
    return loss


def msssim_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    channel: int = None,
    fallback_to_ssim: bool = True,
    return_similarity: bool = False,
) -> torch.Tensor:
    """
    Compute MS-SSIM loss (1 - MS_SSIM).

    Forces FP32 computation for numerical stability under autocast.
    Falls back to a scale-normalized SSIM value below 176px because the
    bundled five-level pytorch_msssim implementation cannot run there.

    Returns:
        Loss tensor [B] (always FP32)
    """
    if channel is None:
        channel = target.shape[1]

    resolution = target.shape[2]

    device_type = "cuda" if target.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        target_f = target.float()
        pred_f = pred.float()

        if resolution < MIN_MSSSIM_RESOLUTION:
            if fallback_to_ssim:
                # ssim_loss already contains the legacy 10x DSSIM scaling.
                # reconstruction_loss applies `weight` once more around this
                # function, so divide here to apply the requested weight once
                # at low resolutions.
                fallback_loss, fallback_similarity = ssim_loss(
                    target_f,
                    pred_f,
                    channel,
                    resolution,
                    return_similarity=True,
                )
                loss = fallback_loss / 10.0
                if return_similarity:
                    return loss, fallback_similarity
                return loss
            raise ValueError(
                f"MS-SSIM requires resolution >= {MIN_MSSSIM_RESOLUTION}; "
                f"received {resolution}."
            )
        ssim_val = SSIMCache.get_msssim(channel)(target_f, pred_f)
        loss = 1.0 - ssim_val

    if return_similarity:
        return loss, ssim_val
    return loss


def msssim_l1_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    channel: int = None,
    alpha: float = 0.84,
    return_similarity: bool = False,
) -> torch.Tensor:
    """
    Compute MS-SSIM + L1 combined loss.

    Forces FP32 computation for numerical stability under BF16 autocast.
    loss = alpha * (1 - MS_SSIM) + (1 - alpha) * L1

    Returns:
        Loss tensor [B] (always FP32)
    """
    if channel is None:
        channel = target.shape[1]

    resolution = target.shape[2]

    device_type = "cuda" if target.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        target_f = target.float()
        pred_f = pred.float()

        l1 = torch.abs(target_f - pred_f).mean(dim=[1, 2, 3])

        if resolution < MIN_MSSSIM_RESOLUTION:
            # Keep the structural component on the same scale as real
            # MS-SSIM before reconstruction_loss applies its outer 10x weight.
            fallback_loss, fallback_similarity = ssim_loss(
                target_f,
                pred_f,
                channel,
                resolution,
                return_similarity=True,
            )
            normalized_ssim = fallback_loss / 10.0
            loss = alpha * normalized_ssim + (1.0 - alpha) * l1
            if return_similarity:
                return loss, fallback_similarity
            return loss
        ssim_val = SSIMCache.get_msssim(channel)(target_f, pred_f)
        loss = alpha * (1.0 - ssim_val) + (1.0 - alpha) * l1

    if return_similarity:
        return loss, ssim_val
    return loss


def reconstruction_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    loss_type: str = "SSIM",
    channel: int = None,
    weight: float = 10.0,
    include_l2: bool = True,
    return_similarity: bool = False,
) -> torch.Tensor:
    """
    Compute reconstruction loss with configurable loss type.

    For SSIM mode, uses resolution-adaptive dual-scale DSSIM matching legacy TF.
    The SSIM path already includes weight scaling internally (5+5 or 10).

    Forces FP32 computation for numerical stability under BF16 autocast.

    Returns:
        Loss tensor [B] (always FP32)
    """
    if channel is None:
        channel = target.shape[1]

    device_type = "cuda" if target.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        target_f = target.float()
        pred_f = pred.float()

        if loss_type == "MS-SSIM":
            structural_loss, similarity = msssim_loss(
                target_f,
                pred_f,
                channel,
                return_similarity=True,
            )
            loss = weight * structural_loss
            if include_l2:
                loss = loss + weight * (target_f - pred_f).pow(2).mean(dim=[1, 2, 3])
        elif loss_type == "MS-SSIM+L1":
            structural_loss, similarity = msssim_l1_loss(
                target_f,
                pred_f,
                channel,
                return_similarity=True,
            )
            loss = weight * structural_loss
        else:  # SSIM
            loss, similarity = ssim_loss(
                target_f,
                pred_f,
                channel,
                return_similarity=True,
            )
            if include_l2:
                loss = loss + weight * (target_f - pred_f).pow(2).mean(dim=[1, 2, 3])

    if return_similarity:
        return loss, similarity
    return loss
