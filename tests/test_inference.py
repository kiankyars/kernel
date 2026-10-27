"""Integration tests compare complete autoregressive behavior, not just shapes."""

import pytest
import torch
import json

from arena import inference
from arena.spec import CandidateSpec


def test_complete_autoregressive_rollout():
    result = inference.integration_check(CandidateSpec(), seed=31, steps=3, repeats=1)
    assert result["passed"]
    assert result["tokens_match"] and result["logits_match"]
    assert len(result["candidate_tokens"]) == 3
    assert result["full_logit_tensors_checked"] == 3
    assert result["convolution_calls_per_rollout"] == 12
    assert result["baseline_latency_ms"] > 0
    assert result["candidate_latency_ms"] > 0
    assert 0 < result["baseline_convolution_fraction"] < 1
    assert result["timing_kind"] == "cpu_interpreter_diagnostic"
    assert not result["production_inference_verified"]


def test_wrong_convolution_fails_before_benchmark(monkeypatch):
    def wrong(hidden, dynamic, base, group_size, spec):
        return torch.zeros_like(hidden)

    monkeypatch.setattr(inference.kernels, "convolve", wrong)
    result = inference.integration_check(CandidateSpec(), seed=31, steps=2, repeats=1)
    assert not result["passed"]
    assert not result["logits_match"]
    assert result["baseline_latency_ms"] is None
    assert result["candidate_latency_ms"] is None
    assert result["speedup"] is None


def test_rollout_is_reproducible():
    first = inference.integration_check(CandidateSpec(), seed=92, steps=2, repeats=1)
    second = inference.integration_check(CandidateSpec(), seed=92, steps=2, repeats=1)
    assert first["passed"] and second["passed"]
    assert first["candidate_tokens"] == second["candidate_tokens"]
    assert first["logits_max_error"] == second["logits_max_error"]


def test_nonfinite_output_is_rejected_with_json_safe_report(monkeypatch):
    def wrong(hidden, dynamic, base, group_size, spec):
        return torch.full_like(hidden, float("nan"))

    monkeypatch.setattr(inference.kernels, "convolve", wrong)
    result = inference.integration_check(CandidateSpec(), steps=1, repeats=1)
    assert not result["passed"]
    assert result["logits_max_error"] is None
    assert result["candidate_latency_ms"] is None
    json.dumps(result, allow_nan=False)


def test_decoder_does_not_read_future_tokens():
    model = inference._TinyDecoder(seed=42, device=torch.device("cpu"))
    left = torch.tensor([[3, 5, 7, 11]])
    right = torch.tensor([[3, 5, 13, 17]])
    left_logits = model(left, inference.reference)
    right_logits = model(right, inference.reference)
    torch.testing.assert_close(left_logits[:, :2], right_logits[:, :2], rtol=0, atol=0)


@pytest.mark.parametrize("steps,repeats", [(0, 1), (65, 1), (2, 0)])
def test_refuses_empty_or_invalid_benchmarks(steps, repeats):
    with pytest.raises(ValueError):
        inference.integration_check(CandidateSpec(), steps=steps, repeats=repeats)
