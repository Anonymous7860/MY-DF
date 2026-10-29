"""
AdaBelief Optimizer - PyTorch Edition

AdaBelief optimizer with additional features:
- Bias correction for stable early training (disabled by default to match TF)
- Linear warmup schedule
- Learning rate dropout (random dropout of parameter updates)
- Cosine annealing learning rate schedule
- Gradient norm clipping

Reference: "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
https://arxiv.org/abs/2010.07468
"""

import math

import numpy as np
import torch

from core.leras import nn
from core.leras.optimizers.OptimizerBase import LrDropoutMode
from core.leras.optimizers.optimizer_factory import OptimizerRegistry



@OptimizerRegistry.register("adabelief")
class AdaBelief(nn.OptimizerBase):
    """
    AdaBelief Optimizer with legacy DFL features.

    AdaBelief adapts the step size according to the "belief" in the gradient direction.
    It views the exponential moving average (EMA) of the noisy gradient as the prediction
    of the gradient at the next time step.

    Legacy Features:
    - lr_dropout: Probability of keeping the update (0.0 to 1.0). If < 1.0, updates
      are randomly skipped for parameters.
    - lr_cos_period: Period for cosine annealing of learning rate in iterations.
      If > 0, LR cycles from lr to 0 and back.
    - clipnorm: Max norm for global gradient clipping. If > 0, gradients are clipped.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate. Default: 1e-3.
        betas: Coefficients for computing running averages of gradient and
            its square. Default: (0.9, 0.999).
        eps: Term added to the denominator to improve numerical stability.
            Default: 1e-16.
        weight_decay: Weight decay (L2 penalty). Default: 0.0.
        lr_dropout: Learning rate dropout probability (keep rate). Default: 1.0 (no dropout).
        lr_cos: Period for cosine LR scheduling. Default: 0 (disabled).
        clipnorm: Maximum global norm for gradient clipping. Default: 0.0 (disabled).
        bias_correction: Enable bias correction. Default: False (matches DFL legacy).
        warmup_steps: Number of linear warmup steps. Default: 0 (disabled).
        weight_decouple: Decouple weight decay (AdamW-style). Default: True.
        name: Optional optimizer name for checkpointing.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        lr_dropout: float = 1.0,
        lr_dropout_mode: str | LrDropoutMode = LrDropoutMode.PER_STEP,
        lr_cos: int = 0,
        clipnorm: float = 0.0,
        bias_correction: bool = False,
        warmup_steps: int = 0,
        name: str | None = None,
        **kwargs,
    ) -> None:
        if name is None:
            name = "AdaBelief"

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            lr_dropout=lr_dropout,
            lr_cos=lr_cos,
            clipnorm=clipnorm,
            bias_correction=bias_correction,
            warmup_steps=warmup_steps,
        )
        super().__init__(
            params,
            defaults,
            name=name,
            lr_dropout_mode=lr_dropout_mode,
        )

        self.lr = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.clipnorm = clipnorm
        self.bias_correction = bias_correction
        self.warmup_steps = warmup_steps
        self.weight_decouple = weight_decouple

    def get_weights(self):
        """
        Get optimizer state tensors for saving.
        Compatible with old TensorFlow API.
        """
        weights = [torch.tensor(self._step_count)]
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    state = self.state[p]
                    if "m" in state:
                        weights.append(state["m"])
                    if "v" in state:
                        weights.append(state["v"])
        return weights

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Uses torch._foreach_* operations to batch ALL parameter updates
        into ~6 fused kernel launches instead of per-parameter Python loops.

        Args:
            closure: A callable that reevaluates the model and returns the loss

        Returns:
            Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            weight_decouple = group.get("weight_decouple", True)
            lr_dropout = group["lr_dropout"]
            lr_cos = group["lr_cos"]
            clipnorm = group["clipnorm"]
            bias_correction = group.get("bias_correction", False)
            warmup_steps = group.get("warmup_steps", 0)

            param_eps = eps if eps is not None else np.finfo(np.float32).resolution

            if clipnorm > 0:
                params_with_grad = [p for p in group["params"] if p.grad is not None]
                self.clip_grad_norm(params_with_grad, clipnorm)

            effective_lr = lr
            if warmup_steps > 0 and self._step_count <= warmup_steps:
                effective_lr = lr * (self._step_count / warmup_steps)
            if lr_cos > 0 and self._step_count > warmup_steps:
                cos_step = self._step_count - warmup_steps
                effective_lr = effective_lr * self.get_lr_scale(lr_cos, cos_step)

            # Collect active parameters and initialize state
            active_params = [p for p in group["params"] if p.grad is not None]
            if not active_params:
                continue

            for p in active_params:
                state = self.state[p]
                if "m_q" in state and "m" not in state:
                    # Checkpoint written by AdaBelief8bit: expand the 8-bit
                    # moments once, then carry on in FP32.
                    from core.leras.optimizers.AdaBelief8bit import dequantize_moments, QUANT_KEYS
                    if state["m_q"].numel() == p.numel():
                        state["m"], state["v"] = dequantize_moments(state, p.shape, p.device)
                    for key in QUANT_KEYS:
                        state.pop(key, None)
                if "m" not in state or state["m"].shape != p.shape:
                    state["m"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                if "v" not in state or state["v"].shape != p.shape:
                    state["v"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)

            # Build lists for foreach operations
            params_f32 = [p.float() for p in active_params]
            grads_f32 = [p.grad.float() for p in active_params]
            ms = [self.state[p]["m"] for p in active_params]
            vs = [self.state[p]["v"] for p in active_params]
            param_dtypes = [p.dtype for p in active_params]

            # L2 regularization (coupled weight decay)
            if weight_decay != 0 and not weight_decouple:
                torch._foreach_add_(grads_f32, params_f32, alpha=weight_decay)

            torch._foreach_lerp_(ms, grads_f32, 1 - beta1)
            grad_residuals = torch._foreach_sub(grads_f32, ms)
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, grad_residuals, grad_residuals, value=1 - beta2)
            del grad_residuals

            if bias_correction:
                bc1 = 1 - beta1 ** self._step_count
                bc2_sqrt = float(np.sqrt(1 - beta2 ** self._step_count))
                update_lr = effective_lr * bc2_sqrt / bc1
                denom_eps = param_eps * bc2_sqrt
            else:
                update_lr = effective_lr
                denom_eps = param_eps

            denoms = torch._foreach_sqrt(vs)
            torch._foreach_add_(denoms, denom_eps)
            lr_dropout_active = lr_dropout < 1.0 and self._lr_dropout_mode != LrDropoutMode.DISABLED
            if not lr_dropout_active and all(dtype == torch.float32 for dtype in param_dtypes):
                if weight_decay != 0 and weight_decouple:
                    torch._foreach_mul_(active_params, 1 - effective_lr * weight_decay)
                torch._foreach_addcdiv_(active_params, ms, denoms, value=-update_lr)
                del denoms
                continue

            updates = torch._foreach_div(ms, denoms)
            del denoms
            torch._foreach_mul_(updates, -update_lr)

            if lr_dropout_active:
                masks = [
                    self.get_lr_dropout_mask(
                        u.shape,
                        lr_dropout,
                        u.device,
                        u.dtype,
                        param_id=self.get_param_identifier(active_params[i]),
                    )
                    for i, u in enumerate(updates)
                ]
                torch._foreach_mul_(updates, masks)

            # Decoupled weight decay (AdamW-style)
            if weight_decay != 0 and weight_decouple:
                torch._foreach_add_(updates, params_f32, alpha=-effective_lr * weight_decay)

            # Apply updates to original parameters
            for i, p in enumerate(active_params):
                p.add_(updates[i].to(param_dtypes[i]))
            del updates

        return loss


nn.AdaBelief = AdaBelief
