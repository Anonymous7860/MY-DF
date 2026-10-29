"""
OptimizerBase - PyTorch Edition

Base class for custom optimizers with gradient clipping utilities.
Extends torch.optim.Optimizer for compatibility with PyTorch's training loop.
"""

import math
from enum import StrEnum

import torch
from torch.optim import Optimizer

from core.leras import nn


class LrDropoutMode(StrEnum):
    DISABLED = "disabled"
    PER_STEP = "per_step"
    FIXED = "fixed"
    FIXED_CPU = "fixed_cpu"


nn.LrDropoutMode = LrDropoutMode


class OptimizerBase(Optimizer):
    """
    Base optimizer class with additional features:
    - Gradient norm clipping
    - Learning rate dropout (random dropout of parameter updates)
    - Cosine annealing learning rate
    - State dict save/load for checkpointing

    Subclasses should implement:
    - _init_state(p): Initialize optimizer state for parameter p
    - _compute_update(p, grad, state, group): Compute parameter update
    """

    def __init__(
        self,
        params,
        defaults,
        name: str | None = None,
        lr_dropout_mode: str | LrDropoutMode = LrDropoutMode.PER_STEP,
        **kwargs,
    ):
        """
        Initialize optimizer.

        Args:
            params: Model parameters to optimize
            defaults: Dictionary of default hyperparameters
            name: Optional name for the optimizer (for checkpointing)
        """
        super().__init__(params, defaults)
        self.name = name
        self._step_count = 0
        if isinstance(lr_dropout_mode, str):
            lr_dropout_mode = LrDropoutMode(lr_dropout_mode)
        self._lr_dropout_mode = lr_dropout_mode
        self._fixed_dropout_masks: dict[int, torch.Tensor] = {}

    @staticmethod
    def clip_grad_norm(parameters, max_norm):
        """
        Clip gradient norm of an iterable of parameters.

        Args:
            parameters: Iterable of parameters with gradients
            max_norm: Maximum norm value (0 means no clipping)

        Returns:
            Total norm of parameter gradients
        """
        if max_norm <= 0:
            return 0.0

        # Calculate total norm
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0

        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)

        total_norm_value = total_norm.item()
        if not math.isfinite(total_norm_value):
            raise FloatingPointError(
                f"Non-finite gradient norm detected before optimizer update: {total_norm_value}"
            )

        # Reuse the synchronized scalar for the comparison and return value.
        clip_coef = max_norm / (total_norm_value + 1e-6)
        if clip_coef < 1:
            gradients = [p.grad.detach() for p in parameters]
            try:
                torch._foreach_mul_(gradients, clip_coef)
            except (RuntimeError, TypeError):
                # Sparse or otherwise unsupported gradient layouts keep the
                # exact legacy elementwise scaling path.
                for gradient in gradients:
                    gradient.mul_(clip_coef)

        return total_norm_value

    def get_lr_scale(self, lr_cos, step):
        """
        Compute learning rate scale for cosine annealing.

        Args:
            lr_cos: Period of cosine annealing (0 = disabled)
            step: Current step count

        Returns:
            Learning rate multiplier (0.0 to 1.0)
        """
        if lr_cos <= 0:
            return 1.0
        return (math.cos(step * 2 * math.pi / lr_cos) + 1.0) / 2.0

    @staticmethod
    def get_param_identifier(param: torch.Tensor) -> int:
        try:
            return int(param.data_ptr())
        except Exception:
            return id(param)

    def get_lr_dropout_mask(self, shape, lr_dropout, device, dtype, param_id=None):
        """
        Generate a dropout mask according to the configured lr_dropout mode.

        Args:
            shape: Shape of the parameter tensor
            lr_dropout: Probability of keeping an update (1.0 = no dropout)
            device: Device to create tensor on
            dtype: Data type of tensor
            param_id: Stable parameter identifier for fixed-mask modes

        Returns:
            Binary mask tensor or 1.0 if no dropout
        """
        mode = self._lr_dropout_mode if self is not None else LrDropoutMode.PER_STEP

        if lr_dropout >= 1.0 or mode == LrDropoutMode.DISABLED:
            return 1.0

        if mode == LrDropoutMode.PER_STEP or self is None:
            return torch.bernoulli(torch.full(shape, lr_dropout, device=device, dtype=dtype))

        if param_id is None:
            return torch.bernoulli(torch.full(shape, lr_dropout, device=device, dtype=dtype))

        if param_id not in self._fixed_dropout_masks:
            mask_device = "cpu" if mode == LrDropoutMode.FIXED_CPU else device
            self._fixed_dropout_masks[param_id] = torch.bernoulli(
                torch.full(shape, lr_dropout, device=mask_device, dtype=dtype)
            )

        mask = self._fixed_dropout_masks[param_id]
        if mask.device != device:
            mask = mask.to(device)
        return mask

    @property
    def lr_dropout_mode(self):
        return self._lr_dropout_mode

    @lr_dropout_mode.setter
    def lr_dropout_mode(self, mode):
        if isinstance(mode, str):
            mode = LrDropoutMode(mode)
        if mode != self._lr_dropout_mode:
            self._fixed_dropout_masks.clear()
        self._lr_dropout_mode = mode

    def clear_fixed_dropout_masks(self):
        self._fixed_dropout_masks.clear()

    def _fresh_lr_dropout_mask(self, shape, lr_dropout, device, dtype):
        return torch.bernoulli(torch.full(shape, lr_dropout, device=device, dtype=dtype))

    def save_state_dict(self, filepath):
        """
        Save optimizer state to file.

        Args:
            filepath: Path to save state
        """
        state = {
            "state": self.state_dict(),
            "step_count": self._step_count,
        }
        if self.name:
            state["name"] = self.name
        torch.save(state, filepath)

    def load_state_dict_from_file(self, filepath, map_location="cpu"):
        """
        Load optimizer state from file.

        Args:
            filepath: Path to load state from
            map_location: Device to map tensors to
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint["state"])
        self._step_count = checkpoint.get("step_count", 0)


nn.OptimizerBase = OptimizerBase
