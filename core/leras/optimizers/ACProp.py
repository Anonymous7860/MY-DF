"""
ACProp Optimizer - PyTorch Edition (foreach-vectorized)

Asynchronous-centering variant of AdaBelief with better convergence properties.
Uses gradient from step t in numerator but v from step t-1 in denominator
(asynchronous update), plus RAdam-style variance rectification.

Convergence rate: O(1/√T) — matches SGD's optimal rate, beats Adam's O(logT/√T).

Reference: "Momentum Centering and Asynchronous Update for Adaptive Gradient Methods"
           Zhuang et al., NeurIPS 2021
           https://arxiv.org/abs/2110.05454

Uses torch._foreach_* operations for batched kernel launches.
"""

import math

import numpy as np
import torch

from core.leras import nn
from core.leras.optimizers.OptimizerBase import LrDropoutMode
from core.leras.optimizers.optimizer_factory import OptimizerRegistry


@OptimizerRegistry.register("acprop")
class ACProp(nn.OptimizerBase):
    """
    ACProp optimizer with foreach-vectorized updates.

    Key differences from AdaBelief:
    1. ASYNCHRONOUS update: denominator uses v from step t-1, not t
    2. RAdam-style rectification: detects when variance estimate is unreliable
       and falls back to SGD-like updates in early training
    3. Better convergence: O(1/√T) vs O(logT/√T) for Adam/AdaBelief

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: None = np.finfo(float32).resolution)
        weight_decay: Weight decay coefficient (default: 0)
        weight_decouple: Decouple weight decay (AdamW-style) (default: True)
        lr_dropout: Probability of keeping parameter updates (default: 1.0)
        lr_cos: Period for cosine annealing (default: 0 = disabled)
        clipnorm: Maximum gradient norm (default: 0.0 = no clipping)
        bias_correction: Enable bias correction (default: True for ACProp)
        warmup_steps: Number of linear warmup steps (default: 0)
        rectify: Use RAdam-style variance rectification (default: True)
        name: Optional optimizer name for checkpointing
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
        bias_correction: bool = True,
        warmup_steps: int = 0,
        rectify: bool = True,
        name: str | None = None,
        **kwargs,
    ) -> None:
        if name is None:
            name = "ACProp"

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
            rectify=rectify,
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
        self.rectify = rectify

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

    def _compute_rho(self, beta2, step):
        """Compute RAdam variance rectification term."""
        beta2_t = beta2 ** step
        rho_max = 2.0 / (1.0 - beta2) - 1.0
        rho = rho_max - 2.0 * step * beta2_t / (1.0 - beta2_t)
        return rho, rho_max

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        [ACProp] Asynchronous update with RAdam rectification:
        - Step 1: Update m (momentum) with current gradient
        - Step 2: Update params using OLD v (from previous step) — ASYNC
        - Step 3: Update v with new gradient residual — for NEXT step's denominator

        Uses torch._foreach_* for batched kernel launches.
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
            warmup_steps = group.get("warmup_steps", 0)
            rectify = group.get("rectify", True)

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

            # Weight decay
            if weight_decay != 0 and not weight_decouple:
                torch._foreach_add_(grads_f32, params_f32, alpha=weight_decay)

            # ===== STEP 1: Update momentum m =====
            # m = beta1 * m + (1 - beta1) * grad
            torch._foreach_mul_(ms, beta1)
            torch._foreach_add_(ms, grads_f32, alpha=1 - beta1)

            # Bias correction for m
            bc1 = 1 - beta1 ** self._step_count

            # ===== STEP 2: ASYNC UPDATE — use OLD v (from step t-1) =====
            # ACProp divides the momentum numerator by the previous second moment.
            # Paper algorithm: Δ_t = m_t / sqrt(v_{t-1} + ε)
            # Reference: https://github.com/juntang-zhuang/ACProp-Optimizer
            if self._step_count <= 1:
                # First step: no v history yet, just use momentum
                updates = torch._foreach_div(ms, bc1)
                torch._foreach_mul_(updates, -effective_lr)
            elif rectify:
                # RAdam-style rectification
                rho, rho_max = self._compute_rho(beta2, self._step_count)

                if rho >= 5:
                    # Variance is reliable — use adaptive step
                    # step_size = rect / bc1 (matches reference exactly)
                    step_size = math.sqrt(
                        (1 - beta2 ** self._step_count)
                        * (rho - 4) / (rho_max - 4)
                        * (rho - 2) / rho
                        * rho_max / (rho_max - 2)
                    ) / bc1

                    # denom = sqrt(v) + eps  (v is from previous step = async)
                    denoms = torch._foreach_sqrt(vs)
                    torch._foreach_add_(denoms, param_eps)

                    # The update numerator is momentum.
                    # Reference: p.data.addcdiv_(exp_avg, denom, value=-step_size * lr)
                    updates = torch._foreach_div(ms, denoms)
                    del denoms
                    torch._foreach_mul_(updates, -effective_lr * step_size)
                else:
                    # Variance unreliable — fall back to SGD-like
                    step_size = 1.0 / bc1
                    updates = [m.clone() for m in ms]
                    torch._foreach_mul_(updates, -effective_lr * step_size)
            else:
                # Non-rectified ACProp: simple async update
                bc2_async = 1 - beta2 ** (self._step_count - 1) if self._step_count > 1 else 1.0

                denoms = torch._foreach_sqrt(vs)
                torch._foreach_div_(denoms, math.sqrt(bc2_async))
                torch._foreach_add_(denoms, param_eps)

                # The update numerator is momentum.
                # Reference: p.data.addcdiv_(exp_avg, denom, value=-step_size)
                updates = torch._foreach_div(ms, denoms)
                del denoms
                step_size = effective_lr / bc1
                torch._foreach_mul_(updates, -step_size)

            # LR dropout
            if lr_dropout < 1.0 and self._lr_dropout_mode != LrDropoutMode.DISABLED:
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

            # Apply updates
            for i, p in enumerate(active_params):
                p.add_(updates[i].to(param_dtypes[i]))
            del updates

            # ===== STEP 3: Update v AFTER param update (async) =====
            # grad_residual = grad - m
            grad_residuals = torch._foreach_sub(grads_f32, ms)
            torch._foreach_mul_(grad_residuals, grad_residuals)

            # v = beta2 * v + (1 - beta2) * residual^2
            torch._foreach_mul_(vs, beta2)
            torch._foreach_add_(vs, grad_residuals, alpha=1 - beta2)
            del grad_residuals

        return loss


nn.ACProp = ACProp
