"""DFlash2 grouped dynamic causal convolution in Triton.

The reference's low-precision operation boundaries are intentional: cast the
base to hidden dtype, round its product, round the first addition, then perform
addcmul in FP32 opmath and round again. Causally padded values are *multiplied*
by weights even when zero, preserving 0*Inf/NaN behavior.

CPU execution uses Triton's actual interpreter: set TRITON_INTERPRET=1 before
importing this module. Interpreter elapsed time is not GPU performance evidence.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from .spec import CandidateSpec


@triton.jit
def _round(value, dtype: tl.constexpr):
    # Triton 3.6's CPU interpreter truncates ordinary float32->bfloat16 casts.
    # Explicit IEEE round-to-nearest-even is identical on CPU and native GPU.
    # Keep the rounded value in FP32 so subsequent arithmetic is always opmath.
    if dtype == tl.bfloat16:
        bits = value.to(tl.uint32, bitcast=True)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        nan_bits = (bits | 0x00400000) & 0xFFFF0000
        rounded = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, nan_bits, rounded)
        return rounded.to(tl.float32, bitcast=True)
    else:
        return value.to(dtype).to(tl.float32)


@triton.jit
def _load_float(value):
    # Bit extension also preserves BF16 subnormals and NaNs in interpretation.
    if value.dtype == tl.bfloat16:
        return (value.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    else:
        return value.to(tl.float32)


@triton.jit
def _step(Hidden, Dynamic, Base, acc, batch, row, channel, valid,
          offset, L: tl.constexpr, GROUP: tl.constexpr,
          HB: tl.constexpr, HL: tl.constexpr, HH: tl.constexpr,
          DB: tl.constexpr, DL: tl.constexpr, DK: tl.constexpr, DG: tl.constexpr,
          BK: tl.constexpr, BH: tl.constexpr, BROKEN: tl.constexpr):
    dtype: tl.constexpr = Hidden.dtype.element_ty
    source_row = row - offset
    if BROKEN:
        source_row += 1
    value = _load_float(tl.load(Hidden + batch * HB + source_row * HL + channel * HH,
                    mask=valid & (source_row >= 0) & (source_row < L), other=0))
    base = tl.load(Base + offset * BK + channel * BH, mask=valid, other=0)
    base = _round(_load_float(base), dtype)
    dynamic = _load_float(tl.load(Dynamic + batch * DB + row * DL + offset * DK
                      + (channel // GROUP) * DG, mask=valid, other=0))
    product = _round(base * value, dtype)
    first_sum = _round(acc + product, dtype)
    return _round(tl.fma(dynamic, value, first_sum), dtype)


@triton.jit
def _fused(Hidden, Dynamic, Base, Out,
           B: tl.constexpr, L: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
           GROUP: tl.constexpr,
           HB: tl.constexpr, HL: tl.constexpr, HH: tl.constexpr,
           DB: tl.constexpr, DL: tl.constexpr, DK: tl.constexpr, DG: tl.constexpr,
           BK: tl.constexpr, BH: tl.constexpr,
           OB: tl.constexpr, OL: tl.constexpr, OH: tl.constexpr,
           BLOCK: tl.constexpr, UNROLL: tl.constexpr, BROKEN: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = index < B * L * H
    channel = index % H
    row = (index // H) % L
    batch = index // (L * H)
    acc = tl.full((BLOCK,), 0, tl.float32)
    if UNROLL:
        for offset in tl.static_range(K):
            acc = _step(Hidden, Dynamic, Base, acc, batch, row, channel, valid,
                        offset, L, GROUP, HB, HL, HH, DB, DL, DK, DG, BK, BH, BROKEN)
    else:
        for offset in range(K):
            acc = _step(Hidden, Dynamic, Base, acc, batch, row, channel, valid,
                        offset, L, GROUP, HB, HL, HH, DB, DL, DK, DG, BK, BH, BROKEN)
    tl.store(Out + batch * OB + row * OL + channel * OH, acc, mask=valid)


@triton.jit
def _split_step(Hidden, Dynamic, Base, Out,
                B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
                GROUP: tl.constexpr, OFFSET: tl.constexpr,
                HB: tl.constexpr, HL: tl.constexpr, HH: tl.constexpr,
                DB: tl.constexpr, DL: tl.constexpr, DK: tl.constexpr, DG: tl.constexpr,
                BK: tl.constexpr, BH: tl.constexpr,
                OB: tl.constexpr, OL: tl.constexpr, OH: tl.constexpr,
                BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = index < B * L * H
    channel = index % H
    row = (index // H) % L
    batch = index // (L * H)
    out_ptr = Out + batch * OB + row * OL + channel * OH
    if OFFSET == 0:
        acc = tl.full((BLOCK,), 0, tl.float32)
    else:
        acc = _load_float(tl.load(out_ptr, mask=valid, other=0))
    acc = _step(Hidden, Dynamic, Base, acc, batch, row, channel, valid,
                OFFSET, L, GROUP, HB, HL, HH, DB, DL, DK, DG, BK, BH, False)
    tl.store(out_ptr, acc, mask=valid)


def _nonoverlapping_output(tensor: torch.Tensor) -> bool:
    # A sufficient check for dense, transposed and gapped strided layouts.
    # Reject ambiguous layouts rather than launching racing stores.
    span = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size <= 1:
            continue
        if stride < span:
            return False
        span += (size - 1) * stride
    return True


def _validate(hidden: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor,
              group_size: int, out: torch.Tensor | None) -> None:
    for name, tensor in (("hidden", hidden), ("dynamic", dynamic), ("base", base)):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"{name} must be a strided torch.Tensor")
        if tensor.requires_grad:
            raise ValueError("this inference kernel does not provide autograd")
        if any(stride < 0 for stride in tensor.stride()):
            raise ValueError("input strides must be nonnegative")
    if hidden.ndim != 3 or dynamic.ndim != 4 or base.ndim != 2:
        raise ValueError("expected hidden[B,L,H], dynamic[B,L,K,H/G], base[K,H]")
    b, length, width = hidden.shape
    if min(b, length, width) < 1:
        raise ValueError("B, L and H must all be nonzero")
    if type(group_size) is not int or group_size < 1 or width % group_size:
        raise ValueError("group_size must be a positive integer dividing H")
    offsets = base.shape[0]
    if not 1 <= offsets <= length + 1:
        raise ValueError("K must be in [1, L+1]")
    if base.shape != (offsets, width) or dynamic.shape != (b, length, offsets, width // group_size):
        raise ValueError("dynamic or base shape does not match hidden/group_size")
    if hidden.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("hidden must be float32, float16, or bfloat16")
    if dynamic.dtype != hidden.dtype or base.dtype not in (hidden.dtype, torch.float32):
        raise ValueError("dynamic must match hidden dtype; base must match or be float32")
    if hidden.device != dynamic.device or hidden.device != base.device:
        raise ValueError("all tensors must be on the same device")
    if hidden.device.type not in ("cpu", "cuda"):
        raise ValueError("only CPU interpretation and CUDA execution are supported")
    if out is not None:
        if not isinstance(out, torch.Tensor) or out.layout != torch.strided:
            raise ValueError("out must be a strided torch.Tensor")
        if out.shape != hidden.shape or out.dtype != hidden.dtype or out.device != hidden.device:
            raise ValueError("out must match hidden shape, dtype and device")
        if out.requires_grad or any(s <= 0 for s in out.stride()) or not _nonoverlapping_output(out):
            raise ValueError("out must have positive nonoverlapping strides and no autograd")
        if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
               for t in (hidden, dynamic, base)):
            raise ValueError("out may not share storage with any input")


def convolve(hidden: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor,
             group_size: int, spec: CandidateSpec = CandidateSpec(),
             out: torch.Tensor | None = None) -> torch.Tensor:
    """Evaluate one bounded candidate, preserving all supplied input tensors.

    Supports positive and broadcast input strides and arbitrary storage offsets.
    An optional nonoverlapping strided ``out`` lets the oracle surround outputs
    with sentinel values. Out-of-contract inputs fail before kernel launch.
    """
    if not isinstance(spec, CandidateSpec):
        raise ValueError("spec must be a CandidateSpec")
    _validate(hidden, dynamic, base, group_size, out)
    if hidden.device.type == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("set TRITON_INTERPRET=1 before importing arena.kernels for CPU execution")
    if out is None:
        out = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    b, length, width = hidden.shape
    offsets = base.shape[0]
    grid = (triton.cdiv(hidden.numel(), spec.block_size),)
    layout = (*hidden.stride(), *dynamic.stride(), *base.stride(), *out.stride())
    launch = {"num_warps": spec.num_warps, "enable_fp_fusion": False}
    if spec.variant == "split":
        for offset in range(offsets):
            _split_step[grid](hidden, dynamic, base, out, b, length, width, group_size,
                              offset, *layout, spec.block_size, **launch)
    else:
        _fused[grid](hidden, dynamic, base, out, b, length, width, offsets, group_size,
                     *layout, spec.block_size, spec.unroll,
                     spec.variant == "broken_shift", **launch)
    return out


def source_digest(spec: CandidateSpec) -> str:
    """Commit a candidate's complete template and typed choices before evaluation."""
    if not isinstance(spec, CandidateSpec):
        raise ValueError("spec must be a CandidateSpec")
    sources = Path(__file__).read_bytes() + Path(__file__).with_name("spec.py").read_bytes()
    choices = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(sources + b"\0" + choices).hexdigest()
