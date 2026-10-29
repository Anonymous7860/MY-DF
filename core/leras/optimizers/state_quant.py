"""Block-wise dynamic 8-bit quantisation for optimizer moment buffers.

Pure PyTorch, no native dependency. Scheme follows the 8-bit optimizer
recipe (Dettmers et al., "8-bit Optimizers via Block-wise Quantization"):

* the tensor is split into blocks of ``BLOCK`` elements; each block is
  normalised by its own absolute maximum (one FP32 scale per block), so one
  outlier cannot flatten the precision of the rest of the tensor;
* the normalised values are mapped to the nearest entry of a 256-entry
  *dynamic* codebook that spends more codes on small magnitudes than a linear
  int8 grid would, which is what the second moment needs (it spans many
  orders of magnitude within one block).

Storage per element: 1 byte of code + 4/BLOCK bytes of scale, i.e. 2.03 bytes
for both AdaBelief moments together instead of 8.0 (BLOCK = 256; measured on
real conv moments, 256 halves the tail error of 2048 for 1.5% more bytes).

The second moment is stored as its square root. Stored directly, elements
more than ~1e7 below their block's maximum round to the zero code, and a
zero second moment collapses the next step's denominator to eps, which
multiplies that element's update by orders of magnitude. In the square-root
domain the same floor sits ~1e14 below the block maximum, and the
dequantised value is squared back, so the denominator never sees a spurious
zero. Measured single-step update-direction agreement with FP32 on trained
conv moments: 0.9944 stored directly vs 0.9998 in the square-root domain.

Only the *storage* is 8-bit. Every optimizer step dequantises a chunk of
parameters to FP32, applies the exact FP32 update, and requantises, so the
math is unchanged and the quantisation error is a small, zero-mean, fresh
rounding of the stored moments each step, not an accumulating drift.
"""
from __future__ import annotations

import torch

BLOCK = 256


def _dynamic_map(signed: bool, max_exponent_bits: int = 7, total_bits: int = 8) -> torch.Tensor:
    """256-entry dynamic codebook in [-1, 1] (signed) or [0, 1] (unsigned).

    Codes are grouped by decade: exponent ``e`` covers magnitudes around
    ``10**(-(max_exponent_bits-1)+e)``, with the fraction bits of each group
    growing as the magnitude grows, so small values keep a usable relative
    resolution and values near the block maximum get the finest steps.
    """
    data: list[float] = []
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
        )
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        scale = 10 ** (-(max_exponent_bits - 1) + i)
        data += (scale * means).tolist()
        if signed:
            data += (-scale * means).tolist()
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        scale = 10 ** (-(max_exponent_bits - 1) + (max_exponent_bits - 1))
        data += (scale * means).tolist()
        if signed:
            data += (-scale * means).tolist()
    data.append(0.0)
    data.append(1.0)
    values = torch.tensor(sorted(set(data)), dtype=torch.float32)
    # Pad to exactly 256 codes with the largest value so any code is valid.
    if values.numel() < 256:
        values = torch.cat([values, values[-1:].expand(256 - values.numel())])
    return values[:256].contiguous()


_CODEBOOKS: dict[tuple[bool, torch.device], torch.Tensor] = {}
_MIDPOINTS: dict[tuple[bool, torch.device], torch.Tensor] = {}


def codebook(signed: bool, device: torch.device) -> torch.Tensor:
    key = (bool(signed), torch.device(device))
    book = _CODEBOOKS.get(key)
    if book is None:
        book = _dynamic_map(signed).to(device)
        _CODEBOOKS[key] = book
    return book


def _midpoints(signed: bool, device: torch.device) -> torch.Tensor:
    key = (bool(signed), torch.device(device))
    mids = _MIDPOINTS.get(key)
    if mids is None:
        book = codebook(signed, device)
        mids = ((book[:-1] + book[1:]) / 2.0).contiguous()
        _MIDPOINTS[key] = mids
    return mids


@torch.no_grad()
def quantize_blockwise(x: torch.Tensor, signed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(codes uint8 [numel], absmax fp32 [blocks])`` for an FP32 tensor."""
    flat = x.detach().reshape(-1).float()
    numel = flat.numel()
    pad = (-numel) % BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, BLOCK)
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    normalised = blocks / absmax.clamp_min(1e-30)
    if not signed:
        normalised = normalised.clamp_(0.0, 1.0)
    codes = torch.bucketize(normalised, _midpoints(signed, x.device), right=False, out_int32=True)
    return codes.to(torch.uint8).reshape(-1)[:numel].contiguous(), absmax.reshape(-1).contiguous()


@torch.no_grad()
def dequantize_blockwise(codes: torch.Tensor, absmax: torch.Tensor, shape, signed: bool, out: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse of :func:`quantize_blockwise`; returns FP32 of ``shape``."""
    numel = int(torch.Size(shape).numel())
    book = codebook(signed, codes.device)
    # index_select accepts int32 indices; skips the int64 widening pass.
    values = torch.index_select(book, 0, codes.reshape(-1).to(torch.int32))
    pad = (-numel) % BLOCK
    if pad:
        values = torch.cat([values, values.new_zeros(pad)])
    values = values.view(-1, BLOCK) * absmax.view(-1, 1)
    values = values.reshape(-1)[:numel].reshape(shape)
    if out is not None:
        out.copy_(values)
        return out
    return values


def quantized_state_bytes(numel: int) -> int:
    """Bytes used by one quantised moment buffer of ``numel`` elements."""
    blocks = (numel + BLOCK - 1) // BLOCK
    return numel + 4 * blocks
