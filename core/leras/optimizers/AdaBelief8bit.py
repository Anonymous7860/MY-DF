"""AdaBelief with block-wise 8-bit moment storage.

Same update rule, hyper-parameters and legacy DFL features as
:class:`core.leras.optimizers.AdaBelief.AdaBelief`; the only difference is
how the two moment buffers live between steps:

    FP32 AdaBelief : state["m"], state["v"]            8.0 bytes / parameter
    this class     : state["m_q"], state["m_scale"],
                     state["v_q"], state["v_scale"]    2.03 bytes / parameter
                     (v_q holds sqrt(v); see state_quant)

Each step processes parameters in chunks: dequantise the chunk's moments to
FP32, run the exact FP32 AdaBelief update, requantise, free. Peak extra
memory is therefore one chunk of FP32 moments, not the whole model's.

State compatibility:
* a saved FP32 AdaBelief state (``m``/``v``) loads into this optimizer and is
  quantised on its first step, so turning the option on keeps momentum;
* a saved 8-bit state loads into plain AdaBelief and is dequantised on its
  first step (see ``AdaBelief.step``), so turning the option off is lossless
  apart from the 8-bit rounding already present.
"""
from __future__ import annotations

import numpy as np
import torch

from core.leras import nn
from core.leras.optimizers.AdaBelief import AdaBelief
from core.leras.optimizers.OptimizerBase import LrDropoutMode
from core.leras.optimizers.optimizer_factory import OptimizerRegistry
from core.leras.optimizers.state_quant import (
    dequantize_blockwise,
    quantize_blockwise,
    quantized_state_bytes,
)

QUANT_KEYS = ("m_q", "m_scale", "v_q", "v_scale")

# Elements of FP32 moments materialised at once (m and v each): 32M -> 256MB peak.
CHUNK_ELEMENTS = 32 * 1024 * 1024


def state_has_quantized_moments(state: dict) -> bool:
    return all(key in state for key in QUANT_KEYS)


def dequantize_moments(state: dict, shape, device) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 (m, v) from a quantised state entry."""
    m = dequantize_blockwise(state["m_q"].to(device), state["m_scale"].to(device), shape, signed=True)
    # v is stored as sqrt(v); see state_quant for why.
    root = dequantize_blockwise(state["v_q"].to(device), state["v_scale"].to(device), shape, signed=False)
    return m, root * root


def quantize_moments_into(state: dict, m: torch.Tensor, v: torch.Tensor) -> None:
    """Store FP32 (m, v) as 8-bit codes, dropping any FP32 copies."""
    state["m_q"], state["m_scale"] = quantize_blockwise(m, signed=True)
    # v is a variance (never negative) and is stored as its square root so
    # its small elements cannot round to zero; see state_quant.
    state["v_q"], state["v_scale"] = quantize_blockwise(v.clamp_min(0.0).sqrt(), signed=False)
    state.pop("m", None)
    state.pop("v", None)


@OptimizerRegistry.register("adabelief8bit")
class AdaBelief8bit(AdaBelief):
    """AdaBelief whose moments are stored block-wise in 8 bits."""

    def __init__(self, params, *args, name: str | None = None, **kwargs):
        super().__init__(params, *args, name=name or "AdaBelief8bit", **kwargs)

    def get_weights(self):
        """Match AdaBelief.get_weights: step counter followed by FP32 m, v per param."""
        weights = [torch.tensor(self._step_count)]
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                if state_has_quantized_moments(state):
                    m, v = dequantize_moments(state, p.shape, p.device)
                    weights += [m, v]
                else:
                    if "m" in state:
                        weights.append(state["m"])
                    if "v" in state:
                        weights.append(state["v"])
        return weights

    def state_bytes(self) -> int:
        """Bytes currently held by the moment storage (for diagnostics)."""
        total = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        total += value.numel() * value.element_size()
        return total

    @torch.no_grad()
    def step(self, closure=None):
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
                effective_lr = effective_lr * self.get_lr_scale(lr_cos, self._step_count - warmup_steps)

            if bias_correction:
                bc1 = 1 - beta1 ** self._step_count
                bc2_sqrt = float(np.sqrt(1 - beta2 ** self._step_count))
                update_lr = effective_lr * bc2_sqrt / bc1
                denom_eps = param_eps * bc2_sqrt
            else:
                update_lr = effective_lr
                denom_eps = param_eps

            active_params = [p for p in group["params"] if p.grad is not None]
            if not active_params:
                continue
            lr_dropout_active = lr_dropout < 1.0 and self._lr_dropout_mode != LrDropoutMode.DISABLED

            # Chunk parameters so only a bounded amount of FP32 moments exists at once.
            chunk: list[torch.Tensor] = []
            chunk_elems = 0
            for p in active_params:
                chunk.append(p)
                chunk_elems += p.numel()
                if chunk_elems >= CHUNK_ELEMENTS:
                    self._step_chunk(chunk, beta1, beta2, denom_eps, update_lr, effective_lr,
                                     weight_decay, weight_decouple, lr_dropout, lr_dropout_active)
                    chunk, chunk_elems = [], 0
            if chunk:
                self._step_chunk(chunk, beta1, beta2, denom_eps, update_lr, effective_lr,
                                 weight_decay, weight_decouple, lr_dropout, lr_dropout_active)

        return loss

    def _materialize(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 (m, v) for parameter ``p``: from 8-bit codes, from a legacy FP32 state, or zeros."""
        state = self.state[p]
        if state_has_quantized_moments(state) and state["m_q"].numel() == p.numel():
            return dequantize_moments(state, p.shape, p.device)
        m = state.get("m")
        v = state.get("v")
        if m is None or m.shape != p.shape:
            m = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
        else:
            m = m.to(device=p.device, dtype=torch.float32)
        if v is None or v.shape != p.shape:
            v = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
        else:
            v = v.to(device=p.device, dtype=torch.float32)
        return m, v

    def _step_chunk(self, params, beta1, beta2, denom_eps, update_lr, effective_lr,
                    weight_decay, weight_decouple, lr_dropout, lr_dropout_active):
        ms, vs = [], []
        for p in params:
            m, v = self._materialize(p)
            ms.append(m)
            vs.append(v)
        params_f32 = [p.float() for p in params]
        grads_f32 = [p.grad.float() for p in params]
        param_dtypes = [p.dtype for p in params]

        if weight_decay != 0 and not weight_decouple:
            torch._foreach_add_(grads_f32, params_f32, alpha=weight_decay)

        torch._foreach_lerp_(ms, grads_f32, 1 - beta1)
        grad_residuals = torch._foreach_sub(grads_f32, ms)
        torch._foreach_mul_(vs, beta2)
        torch._foreach_addcmul_(vs, grad_residuals, grad_residuals, value=1 - beta2)
        del grad_residuals

        denoms = torch._foreach_sqrt(vs)
        torch._foreach_add_(denoms, denom_eps)
        updates = torch._foreach_div(ms, denoms)
        del denoms
        torch._foreach_mul_(updates, -update_lr)

        if lr_dropout_active:
            masks = [
                self.get_lr_dropout_mask(u.shape, lr_dropout, u.device, u.dtype,
                                         param_id=self.get_param_identifier(params[i]))
                for i, u in enumerate(updates)
            ]
            torch._foreach_mul_(updates, masks)

        if weight_decay != 0 and weight_decouple:
            torch._foreach_add_(updates, params_f32, alpha=-effective_lr * weight_decay)

        for i, p in enumerate(params):
            p.add_(updates[i].to(param_dtypes[i]))
            quantize_moments_into(self.state[p], ms[i], vs[i])
        del updates, ms, vs


nn.AdaBelief8bit = AdaBelief8bit
