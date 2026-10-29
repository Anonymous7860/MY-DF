"""
Muon Optimizer - PyTorch Edition

MomentUm Orthogonalized by Newton-Schulz (Keller Jordan et al., 2024), with
Moonlight RMS-matched update scaling (Liu et al. 2025) so a SINGLE Adam-scale
learning rate drives both the Muon and AdamW groups — no separate Muon LR.
- Newton-Schulz coeffs (3.4445, -4.7750, 2.0315), 5 steps, bfloat16.
- momentum 0.95, nesterov True.
- update scale = 0.2 * sqrt(max(out, in))  (RMS-matched to AdamW).

Routing via the official `MuonWithAuxAdam` recipe (Muon = hidden weight layers;
AdamW = the input embedding, final output layer, and all gains/biases):
- Matrix-shaped hidden params (ndim >= 2) get the Muon orthogonalized update.
- Put embedding/head params in a group with 'use_muon': False to route them to
  AdamW; 1-D params always take the AdamW path regardless of group flag.

Supports the SAENXT trainer conventions: per-group weight decay, decoupled
weight decay, LR dropout, cosine LR, warmup, global grad-norm clipping.
"""

import math

import torch

from core.leras import nn
from core.leras.optimizers.OptimizerBase import LrDropoutMode
from core.leras.optimizers.optimizer_factory import OptimizerRegistry


def _newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
    """Approximately orthogonalize G via the quintic Newton-Schulz iteration.

    Coefficients from Keller Jordan's reference implementation; computed in
    bfloat16 on CUDA (the iteration is robust to low precision) for speed.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16) if G.is_cuda else G.float()
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.float()


@OptimizerRegistry.register("muon")
class Muon(nn.OptimizerBase):
    """
    Muon with AdamW fallback for non-matrix parameters.

    Args:
        params: Iterable of parameters or param-group dicts.
        lr: Learning rate (single Adam-scale rate for both paths, thanks to the
            RMS-matched update scaling).
        betas: AdamW betas for the fallback path. Default: (0.9, 0.999).
        momentum: Muon momentum. Default: 0.95.
        nesterov: Nesterov momentum on the Muon path. Default: True.
        ns_steps: Newton-Schulz iterations. Default: 5.
        eps: AdamW epsilon. Default: 1e-8.
        weight_decay: Decoupled weight decay. Default: 0.0 (per-group override).
        weight_decouple: Kept for trainer API compatibility (always decoupled).
        lr_dropout / lr_dropout_mode / lr_cos / clipnorm / warmup_steps:
            Same semantics as the other SAENXT optimizers.
    """

    def __init__(
        self,
        params,
        lr: float = 5e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        lr_dropout: float = 1.0,
        lr_dropout_mode: str | LrDropoutMode = LrDropoutMode.PER_STEP,
        lr_cos: int = 0,
        clipnorm: float = 0.0,
        warmup_steps: int = 0,
        name: str | None = None,
        **kwargs,
    ) -> None:
        if name is None:
            name = "Muon"

        defaults = dict(
            lr=lr,
            betas=betas,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            lr_dropout=lr_dropout,
            lr_cos=lr_cos,
            clipnorm=clipnorm,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults, name=name, lr_dropout_mode=lr_dropout_mode)

        self.lr = lr
        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.clipnorm = clipnorm
        self.warmup_steps = warmup_steps
        self.eps = eps

    def get_weights(self):
        """Compatible with old TensorFlow API / checkpointing."""
        weights = [torch.tensor(self._step_count)]
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    state = self.state[p]
                    for key in ("momentum_buffer", "m", "v"):
                        if key in state:
                            weights.append(state[key])
        return weights

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        t = self._step_count

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            momentum = group.get("momentum", 0.95)
            nesterov = group.get("nesterov", True)
            ns_steps = group.get("ns_steps", 5)
            eps = group.get("eps", self.eps)
            weight_decay = group.get("weight_decay", 0.0)
            lr_dropout = group.get("lr_dropout", 1.0)
            lr_cos = group.get("lr_cos", 0)
            clipnorm = group.get("clipnorm", 0.0)
            warmup_steps = group.get("warmup_steps", 0)

            # Official MuonWithAuxAdam: a group may set use_muon=False to run
            # entirely on the AdamW path (input embedding, final output layer,
            # gains/biases). Default True.
            use_muon = group.get("use_muon", True)

            active_params = [p for p in group["params"] if p.grad is not None]
            if not active_params:
                continue

            if clipnorm > 0:
                self.clip_grad_norm(active_params, clipnorm)

            effective_lr = lr
            if warmup_steps > 0 and t <= warmup_steps:
                effective_lr = lr * (t / warmup_steps)
            if lr_cos > 0 and t > warmup_steps:
                effective_lr = effective_lr * self.get_lr_scale(lr_cos, t - warmup_steps)

            def _apply_update(p, update):
                """Dropout mask, decoupled decay, and parameter update.

                Shared tail of both paths; op order matches the original
                per-parameter loop exactly (mask -> decay -> add)."""
                if lr_dropout < 1.0 and self._lr_dropout_mode != LrDropoutMode.DISABLED:
                    mask = self.get_lr_dropout_mask(
                        update.shape,
                        lr_dropout,
                        update.device,
                        update.dtype,
                        param_id=self.get_param_identifier(p),
                    )
                    update = update * mask

                if weight_decay != 0:
                    # Decoupled (AdamW-style) weight decay on both paths.
                    p.mul_(1.0 - effective_lr * weight_decay)

                p.add_(update.to(p.dtype), alpha=-effective_lr)

            # ---- AdamW fallback (biases, norm gains, scalars), batched ----
            # torch._foreach_ ops run identical elementwise math to the old
            # per-parameter loop but fuse it into a handful of kernel
            # launches. The two bias-correction divisions stay per-tensor:
            # CUDA's foreach divide-by-scalar overloads pick
            # reciprocal-multiply or true division depending on the scalar's
            # value, so none of them bitwise-match `m / bc1` for every step.
            adamw_path_params = [
                p for p in active_params if not (use_muon and p.ndim >= 2)
            ]
            adamw_updates = {}
            params_by_device = {}
            for p in adamw_path_params:
                params_by_device.setdefault(p.device, []).append(p)
            for device_params in params_by_device.values():
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                for p in device_params:
                    state = self.state[p]
                    if "m" not in state or state["m"].shape != p.shape:
                        state["m"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                        state["v"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                    grads.append(p.grad.float())
                    exp_avgs.append(state["m"])
                    exp_avg_sqs.append(state["v"])

                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                # update = (m / bc1) / ((v / bc2).sqrt() + eps)
                numerators = [m / bc1 for m in exp_avgs]
                denominators = torch._foreach_sqrt(
                    [v / bc2 for v in exp_avg_sqs]
                )
                torch._foreach_add_(denominators, eps)
                updates = torch._foreach_div(numerators, denominators)
                for p, update in zip(device_params, updates):
                    adamw_updates[p] = update

            # Apply updates in the ORIGINAL parameter order: per-step
            # lr_dropout masks draw from the global RNG per parameter, so the
            # draw order must match the old interleaved loop exactly.
            for p in active_params:
                if use_muon and p.ndim >= 2:
                    # ---- Muon path (hidden weight matrices / conv kernels) ----
                    grad = p.grad.float()
                    state = self.state[p]
                    if "momentum_buffer" not in state or state["momentum_buffer"].shape != p.shape:
                        state["momentum_buffer"] = torch.zeros(
                            p.shape, dtype=torch.float32, device=p.device
                        )
                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1.0 - momentum)
                    g = grad.lerp(buf, momentum) if nesterov else buf

                    g2d = g.reshape(g.shape[0], -1)
                    ortho = _newton_schulz_orthogonalize(g2d, steps=ns_steps)
                    # Moonlight RMS matching (Liu et al. 2025): scales the
                    # orthogonalized update so its RMS matches AdamW's, letting a
                    # single Adam-scale LR drive both the Muon and AdamW groups.
                    scale = 0.2 * math.sqrt(max(g2d.shape[0], g2d.shape[1]))
                    update = ortho.reshape(p.shape) * scale
                else:
                    update = adamw_updates[p]
                _apply_update(p, update)

        return loss


nn.Muon = Muon
