"""Tests exercise oracle semantics, hidden-case generation, and rejection gates."""

import pytest
import torch

from arena.oracle import Case, assert_equivalent, evaluate, make_cases, reference
from arena.spec import CandidateSpec


def test_reference_hand_computed_causal_grouping():
    hidden = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.]]])
    dynamic = torch.tensor([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]])
    base = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    # t=0 sees offset zero only; t=1 additionally uses the preceding hidden row.
    expected = torch.tensor([[[2., 6., 15., 24.], [42., 68., 108., 144.]]])
    torch.testing.assert_close(reference(hidden, dynamic, base, 2), expected)


def test_reference_all_padding_offset_preserves_nan_semantics():
    hidden = torch.ones(1, 1, 1)
    dynamic = torch.ones(1, 1, 2, 1)
    base = torch.tensor([[1.], [float("inf")]])
    assert torch.isnan(reference(hidden, dynamic, base, 1)).all()


def test_cases_repeatable_with_noncontiguous_storage():
    first, second = make_cases(541, 24), make_cases(541, 24)
    for a, b in zip(first, second):
        assert a.name == b.name
        assert a.group_size == b.group_size
        for ta, tb in zip((a.hidden, a.dynamic, a.base), (b.hidden, b.dynamic, b.base)):
            assert ta.stride() == tb.stride()
            assert ta.storage_offset() == tb.storage_offset()
            torch.testing.assert_close(ta, tb, rtol=0, atol=0, equal_nan=True)
    assert any(not case.hidden.is_contiguous() for case in first)
    assert {case.hidden.dtype for case in first} == {torch.float32, torch.float16, torch.bfloat16}


def test_zero_cases_rejected():
    with pytest.raises(ValueError, match="zero tests"):
        make_cases(0, 0)
    result = evaluate(CandidateSpec(), [])
    assert not result["passed"]
    assert result["cases_run"] == 0
    assert result["failures"]


@pytest.mark.parametrize("variant", ["fused", "split"])
def test_real_kernel_passes_oracle(variant):
    result = evaluate(CandidateSpec(variant=variant), make_cases(17, 24))
    assert result["passed"], result
    assert result["cases_run"] == 24
    assert not result["failures"]


def test_deliberate_incorrect_mutation_rejected():
    result = evaluate(CandidateSpec(variant="broken_shift"), make_cases(17, 12))
    assert not result["passed"]
    assert result["cases_run"] >= 1
    assert result["failures"]


@pytest.mark.parametrize("actual,expected", [
    (float("nan"), 0.), (0., float("nan")),
    (float("inf"), -float("inf")), (1., float("inf")),
])
def test_special_value_mismatch_is_never_tolerated(actual, expected):
    with pytest.raises(AssertionError):
        assert_equivalent(torch.tensor([actual]), torch.tensor([expected]))


def test_input_backing_storage_mutation_rejected(monkeypatch):
    def corrupt(hidden, dynamic, base, group_size, spec, out):
        out.copy_(reference(hidden, dynamic, base, group_size))
        hidden.as_strided((1,), (1,), 0).fill_(99.)
        return out
    monkeypatch.setattr("arena.kernels.convolve", corrupt)
    result = evaluate(CandidateSpec(), make_cases(17, 1))
    assert not result["passed"]
    assert "input backing storage" in result["failures"][0]["error"]


def test_output_hole_mutation_rejected(monkeypatch):
    def corrupt(hidden, dynamic, base, group_size, spec, out):
        out.copy_(reference(hidden, dynamic, base, group_size))
        out.as_strided((1,), (1,), 0).fill_(99.)
        return out
    monkeypatch.setattr("arena.kernels.convolve", corrupt)
    result = evaluate(CandidateSpec(), make_cases(17, 1))
    assert not result["passed"]
    assert "output sentinel" in result["failures"][0]["error"]
