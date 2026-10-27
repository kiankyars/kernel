"""Independent PyTorch oracle and randomized, guarded correctness cases.

The arithmetic order is the DFlash implementation, not an algebraically
reassociated sum. Source: z-lab/dflash, commit
07ebd93db9f472af339b644bb70221ad8428328a, dflash/model.py:478-489.
The CPU interpreter is a functional oracle, not evidence of GPU speed or races.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    hidden: torch.Tensor
    dynamic: torch.Tensor
    base: torch.Tensor
    group_size: int
    name: str


# Small, explicit rounding allowances; special-value classifications remain
# exact. FP16/BF16 tolerances are not claims of bitwise numerical equivalence.
TOLERANCES = {
    torch.float32: {"atol": 2e-6, "rtol": 2e-5},
    torch.float16: {"atol": 2e-3, "rtol": 2e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
}


def reference(hidden, dynamic, base, group_size):
    """Literal trusted operation, preserving the original rounding boundaries."""
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(
            blocks[:, :-offset], (0, 0, 0, 0, offset, 0)
        )
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


def _guarded_tensor(shape, dtype, generator, rng, device, *, strided=True):
    """Allocate separate backing storage with prefix, suffix, and stride holes."""
    stride = rng.choice([1, 2, 3]) if strided else 1
    strides = [stride]
    for size in reversed(shape[1:]):
        stride = size * stride + (rng.randint(1, 4) if strided else 0)
        strides.insert(0, stride)
    offset = rng.randint(1, 13)
    span = sum((size - 1) * step for size, step in zip(shape, strides)) + 1
    storage = torch.full((offset + span + 17,), -31.25, dtype=dtype, device=device)
    tensor = storage.as_strided(shape, strides, offset)
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    tensor.copy_(values.to(device=device, dtype=dtype))
    return tensor


def make_cases(seed: int, count: int, device="cpu") -> list[Case]:
    """Make reproducible tests; the caller owns and withholds the holdout seed.

    This is not a sandbox for hostile Python. The arena's bounded candidate
    specification cannot read the seed or evaluator files. Public test fixtures
    and user-supplied fixed seeds should never be described as hidden holdouts.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer; zero tests are invalid")
    rng = random.Random(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed % (2**63))
    cases = []
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    for index in range(count):
        dtype = dtypes[index % len(dtypes)]
        batch = rng.choice([1, 2, 3])
        length = rng.choice([1, 2, 3, 7, 9, 17])
        group_size = rng.choice([1, 2, 3, 4, 8])
        groups = rng.choice([1, 2, 3, 7, 11])
        hidden_size = group_size * groups
        kernel_size = rng.choice([1, min(3, length + 1), min(7, length + 1), length + 1])
        # Explicit singleton and causal boundary cases supplement random shapes.
        if index % 12 == 0:
            batch, length, hidden_size, group_size, kernel_size, groups = 1, 1, 1, 1, 2, 1
        elif index % 12 == 1:
            batch, length, hidden_size, group_size, kernel_size, groups = 2, 3, 21, 3, 4, 7
        strided = index % 2 == 1
        hidden = _guarded_tensor((batch, length, hidden_size), dtype, generator, rng, device, strided=strided)
        dynamic = _guarded_tensor((batch, length, kernel_size, groups), dtype, generator, rng, device, strided=strided)
        base_dtype = torch.float32 if index % 3 == 1 else dtype
        base = _guarded_tensor((kernel_size, hidden_size), base_dtype, generator, rng, device, strided=strided)
        mode = (index // 3) % 8
        if mode == 1:
            hidden.zero_()
        elif mode == 2:
            hidden.mul_(0.001)
            base.mul_(100)
        elif mode == 3:
            hidden.mul_(100)
            dynamic.mul_(0.01)
        elif mode == 4:
            hidden[0, 0, 0] = float("nan")
        elif mode == 5:
            hidden[0, 0, 0] = float("inf")
            if hidden_size > 1:
                hidden[0, -1, 1] = -float("inf")
        elif mode == 6:
            # The padded source value is zero: 0*Inf must still produce NaN.
            dynamic[0, 0, kernel_size - 1, 0] = float("inf")
        elif mode == 7:
            base[-1, 0] = float("nan")
        name = f"case-{index}-{str(dtype).split('.')[-1]}-mode{mode}-{'strided' if strided else 'dense'}"
        cases.append(Case(hidden, dynamic, base, group_size, name))
    return cases


def _storage(tensor):
    elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    return tensor.as_strided((elements,), (1,), 0)


def _storage_bytes(tensor):
    return _storage(tensor).view(torch.uint8)


def _outside_view_mask(tensor):
    mask = torch.ones_like(_storage(tensor), dtype=torch.bool)
    logical = mask.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())
    logical.fill_(False)
    return mask


def assert_equivalent(actual, expected):
    """Check special values exactly, then finite values at declared precision."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("output shape or dtype differs from the reference")
    if not torch.equal(torch.isnan(actual), torch.isnan(expected)):
        raise AssertionError("NaN locations differ")
    if not torch.equal(torch.isposinf(actual), torch.isposinf(expected)):
        raise AssertionError("positive-infinity locations differ")
    if not torch.equal(torch.isneginf(actual), torch.isneginf(expected)):
        raise AssertionError("negative-infinity locations differ")
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], **TOLERANCES[expected.dtype])


def evaluate(spec, cases: Iterable[Case]) -> dict:
    """Absolute correctness/mutation gate; never measures candidate performance.

    The first failure stops evaluation and invalidates the candidate. Guard
    checks cover every byte of input backing storage and all output holes.
    They supplement a sanitizer; they cannot detect arbitrary invalid reads.
    """
    from arena.kernels import convolve

    cases = list(cases)
    result = {"passed": False, "cases_run": 0, "cases_total": len(cases), "failures": []}
    if not cases:
        result["failures"].append({"case": None, "error": "zero-test evaluation is invalid"})
        return result
    for index, case in enumerate(cases):
        result["cases_run"] += 1
        try:
            inputs = (case.hidden, case.dynamic, case.base)
            snapshots = [_storage_bytes(t).clone() for t in inputs]
            expected = reference(*inputs, case.group_size)
            out = _guarded_tensor(case.hidden.shape, case.hidden.dtype,
                                  torch.Generator().manual_seed(index), random.Random(index),
                                  case.hidden.device, strided=True)
            outside = _outside_view_mask(out)
            old_output = _storage(out).clone()
            # Poison logical output so a partially unwritten kernel cannot pass.
            out.fill_(float("nan"))
            actual = convolve(*inputs, case.group_size, spec=spec, out=out)
            if actual.data_ptr() != out.data_ptr() or actual.stride() != out.stride():
                raise AssertionError("candidate did not return the provided output view")
            assert_equivalent(actual, expected)
            if not torch.equal(_storage(out)[outside], old_output[outside]):
                raise AssertionError("candidate overwrote output sentinel storage")
            for label, tensor, snapshot in zip(("hidden", "dynamic", "base"), inputs, snapshots):
                if not torch.equal(_storage_bytes(tensor), snapshot):
                    raise AssertionError(f"candidate mutated {label} input backing storage")
        except Exception as exc:
            result["failures"].append({"case": case.name, "error": f"{type(exc).__name__}: {str(exc)[:1500]}"})
            return result
    result["passed"] = True
    return result
