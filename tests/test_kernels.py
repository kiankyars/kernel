"""Kernel API boundaries supplement the randomized numerical/memory oracle."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from arena.kernels import convolve, source_digest
from arena.oracle import assert_equivalent, reference
from arena.spec import CandidateSpec


def _inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(914)
    return (
        torch.randn((2, 3, 4), generator=generator, dtype=dtype),
        torch.randn((2, 3, 2, 2), generator=generator, dtype=dtype),
        torch.randn((2, 4), generator=generator, dtype=torch.float32),
    )


@pytest.mark.parametrize("variant", ["fused", "split"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_broadcast_reads_and_transposed_guarded_output(variant, dtype):
    # All three inputs have zero-stride broadcast dimensions. Inputs are read-only.
    hidden = torch.tensor([1., -2., 3., -4.], dtype=dtype).reshape(1, 1, 4).expand(2, 3, 4)
    dynamic = torch.tensor([.25, -.5], dtype=dtype).reshape(1, 1, 2, 1).expand(2, 3, 2, 2)
    base = torch.tensor([.125, -.25]).reshape(2, 1).expand(2, 4)
    # Logical H is the largest stride, with gaps plus prefix/suffix sentinels.
    backing = torch.full((64,), -37., dtype=dtype)
    out = backing.as_strided((2, 3, 4), (1, 2, 12), storage_offset=7)
    outside = torch.ones((64,), dtype=torch.bool)
    outside.as_strided(out.shape, out.stride(), out.storage_offset()).fill_(False)
    before_inputs = [value.clone() for value in (hidden, dynamic, base)]
    result = convolve(hidden, dynamic, base, 2,
                      spec=CandidateSpec(variant=variant, unroll=False), out=out)
    assert result is out
    assert_equivalent(result, reference(hidden, dynamic, base, 2))
    assert torch.equal(backing[outside], torch.full_like(backing[outside], -37.))
    for actual, before in zip((hidden, dynamic, base), before_inputs):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


@pytest.mark.parametrize("malformation,message", [
    ("dynamic_dtype", "dynamic must match"),
    ("base_dtype", "base must match"),
    ("hidden_dtype", "hidden must be"),
    ("dynamic_shape", "shape does not match"),
    ("hidden_rank", "expected hidden"),
    ("empty_length", "must all be nonzero"),
    ("empty_kernel", "K must be"),
    ("oversized_kernel", "K must be"),
    ("device_mismatch", "same device"),
    ("requires_grad", "autograd"),
    ("sparse_layout", "strided torch.Tensor"),
])
def test_invalid_inputs_rejected_before_launch(malformation, message):
    hidden, dynamic, base = _inputs()
    if malformation == "dynamic_dtype":
        dynamic = dynamic.to(torch.float16)
    elif malformation == "base_dtype":
        base = base.to(torch.float16)
    elif malformation == "hidden_dtype":
        hidden = hidden.to(torch.float64)
    elif malformation == "dynamic_shape":
        dynamic = dynamic[:, :, :, :1]
    elif malformation == "hidden_rank":
        hidden = hidden[0]
    elif malformation == "empty_length":
        hidden = hidden[:, :0]
    elif malformation == "empty_kernel":
        base = base[:0]
    elif malformation == "oversized_kernel":
        base = torch.empty(5, 4)
    elif malformation == "device_mismatch":
        base = torch.empty(base.shape, device="meta")
    elif malformation == "requires_grad":
        hidden.requires_grad_()
    elif malformation == "sparse_layout":
        hidden = hidden.to_sparse()
    with pytest.raises(ValueError, match=message):
        convolve(hidden, dynamic, base, 2)


@pytest.mark.parametrize("group_size", [0, -1, 3, 1.5, True])
def test_invalid_group_size_rejected(group_size):
    with pytest.raises(ValueError, match="group_size"):
        convolve(*_inputs(), group_size)


@pytest.mark.parametrize("bad_out,message", [
    ("hidden_alias", "share storage"),
    ("dynamic_alias", "share storage"),
    ("overlap", "nonoverlapping"),
    ("broadcast", "positive nonoverlapping"),
    ("dtype", "match hidden"),
    ("shape", "match hidden"),
    ("gradient", "autograd"),
])
def test_unsafe_outputs_rejected_before_writes(bad_out, message):
    hidden, dynamic, base = _inputs()
    if bad_out == "hidden_alias":
        out = hidden.view_as(hidden)
    elif bad_out == "dynamic_alias":
        out = dynamic.view_as(hidden)
    elif bad_out == "overlap":
        out = torch.empty(24).as_strided(hidden.shape, (1, 1, 1))
    elif bad_out == "broadcast":
        out = torch.empty(1, 3, 4).expand_as(hidden)
    elif bad_out == "dtype":
        out = torch.empty_like(hidden, dtype=torch.float16)
    elif bad_out == "shape":
        out = torch.empty(1, 3, 4)
    else:
        out = torch.empty_like(hidden, requires_grad=True)
    before = hidden.clone()
    with pytest.raises(ValueError, match=message):
        convolve(hidden, dynamic, base, 2, out=out)
    torch.testing.assert_close(hidden, before, rtol=0, atol=0)


def test_singleton_with_all_padding_offset_and_fp32_base():
    hidden = torch.tensor([[[1.25]]], dtype=torch.bfloat16)
    dynamic = torch.tensor([[[[.5], [2.]]]], dtype=torch.bfloat16)
    base = torch.tensor([[.123456], [.9]], dtype=torch.float32)
    result = convolve(hidden, dynamic, base, 1)
    torch.testing.assert_close(result, reference(hidden, dynamic, base, 1), rtol=0, atol=0)


@pytest.mark.parametrize("arguments", [
    {"block_size": 31}, {"block_size": True}, {"num_warps": 3},
    {"num_warps": 1.0}, {"unroll": 1}, {"variant": "python"},
])
def test_untrusted_spec_values_rejected(arguments):
    with pytest.raises(ValueError):
        CandidateSpec(**arguments)


def test_specs_frozen_and_digest_commits_choices():
    spec = CandidateSpec()
    with pytest.raises(FrozenInstanceError):
        spec.variant = "split"
    assert source_digest(spec) == source_digest(CandidateSpec())
    assert source_digest(spec) != source_digest(CandidateSpec(variant="split"))
    with pytest.raises(ValueError, match="CandidateSpec"):
        convolve(*_inputs(), 2, spec={"variant": "fused"})
