"""Synthetic autoregressive integration, never a production DFlash benchmark.

The random-weight decoder performs causal self-attention, an MLP, residual
connections and four grouped dynamic convolutions per layer, following the
placement in DFlash2. It has no trained checkpoint, tokenizer or KV cache. Its
purpose is to catch a locally correct-looking kernel that changes model outputs
and to measure *this* complete inference workload without hiding other work.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable

import torch
from torch.nn import functional as F

from . import kernels
from .oracle import reference
from .spec import CandidateSpec


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _TinyDecoder:
    """One layer with independent attention/MLP convolution parameters."""

    hidden_size = 32
    vocabulary_size = 64
    group_size = 8
    kernel_size = 2

    def __init__(self, seed: int, device: torch.device):
        self.device = device
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def weight(*shape: int, scale: float = 0.15) -> torch.Tensor:
            return (torch.randn(shape, generator=generator) * scale).to(device)

        h, v, k = self.hidden_size, self.vocabulary_size, self.kernel_size
        g = h // self.group_size
        self.embedding = weight(v, h, scale=0.4)
        self.position = weight(128, h, scale=0.05)
        self.qkv = weight(h, 3 * h, scale=1 / math.sqrt(h))
        self.attention_out = weight(h, h, scale=1 / math.sqrt(h))
        self.mlp_in = weight(h, 2 * h, scale=1 / math.sqrt(h))
        self.mlp_out = weight(2 * h, h, scale=1 / math.sqrt(2 * h))
        self.lm_head = weight(h, v, scale=1 / math.sqrt(h))
        self.dynamic_projections = [weight(h, 2 * k * g, scale=0.06) for _ in range(2)]
        self.bases = [weight(2, k, h, scale=0.2) for _ in range(2)]

    @staticmethod
    def _norm(hidden: torch.Tensor) -> torch.Tensor:
        return hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + 1e-5)

    def __call__(self, ids: torch.Tensor, convolution: Callable) -> torch.Tensor:
        length = ids.shape[1]
        hidden = self.embedding[ids] + self.position[:length]
        for branch in range(2):
            residual = hidden
            hidden = self._norm(hidden)
            dynamic = (hidden @ self.dynamic_projections[branch]).view(
                ids.shape[0], length, 2, self.kernel_size,
                self.hidden_size // self.group_size,
            )
            hidden = convolution(hidden, dynamic[:, :, 0], self.bases[branch][0], self.group_size)
            if branch == 0:
                query, key, value = (hidden @ self.qkv).chunk(3, dim=-1)
                scores = query @ key.transpose(-2, -1) / math.sqrt(self.hidden_size)
                future = torch.ones(length, length, device=self.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(future, -float("inf"))
                hidden = (scores.softmax(dim=-1) @ value) @ self.attention_out
            else:
                hidden = F.silu(hidden @ self.mlp_in) @ self.mlp_out
            hidden = convolution(hidden, dynamic[:, :, 1], self.bases[branch][1], self.group_size)
            hidden = residual + hidden
        return self._norm(hidden) @ self.lm_head


def _decode(model: _TinyDecoder, prompt: torch.Tensor, convolution: Callable, steps: int):
    ids = prompt.clone()
    trace = []
    for _ in range(steps):
        logits = model(ids, convolution)
        trace.append(logits)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, token), dim=1)
    return ids, trace


@torch.inference_mode()
def integration_check(
    spec: CandidateSpec,
    seed: int = 1729,
    device: str = "cpu",
    steps: int = 4,
    repeats: int = 3,
) -> dict:
    """Compare complete greedy rollouts before timing either implementation.

    Timings exclude model construction and compilation warmup, include every
    decode operation, and synchronize CUDA at complete-rollout boundaries.
    CPU interpreter timing is diagnostic and must not be GPU search fitness.
    The convolution fraction comes from a separate instrumented baseline run;
    it is not a production profile and includes per-call instrumentation costs.
    """
    if not 1 <= steps <= 64 or repeats < 1:
        raise ValueError("steps must be in [1, 64] and repeats must be positive")
    target = torch.device(device)
    model = _TinyDecoder(seed, target)
    prompt = torch.tensor([[3, 7, 11, 5, 2, 9, 13, 17]], device=target)

    def candidate(hidden, dynamic, base, group_size):
        return kernels.convolve(hidden, dynamic, base, group_size, spec=spec)

    expected_ids, expected_trace = _decode(model, prompt, reference, steps)
    actual_ids, actual_trace = _decode(model, prompt, candidate, steps)
    tokens_match = torch.equal(expected_ids, actual_ids)
    finite_logits = all(bool(torch.isfinite(logits).all()) for logits in actual_trace)
    differences = [float((a - b).abs().max().item()) for a, b in zip(actual_trace, expected_trace)]
    logits_match = all(
        torch.allclose(actual, expected, rtol=2e-5, atol=2e-5, equal_nan=False)
        for actual, expected in zip(actual_trace, expected_trace)
    )
    result = {
        "passed": bool(tokens_match and logits_match),
        "scope": "synthetic_random_weight_causal_decoder",
        "timing_kind": "cpu_interpreter_diagnostic" if target.type == "cpu" else "native_cuda_synthetic_decoder",
        "production_inference_verified": False,
        "seed": seed,
        "steps": steps,
        "full_logit_tensors_checked": len(expected_trace),
        "logits_match": logits_match,
        "logits_max_error": max(differences) if finite_logits else None,
        "finite_logits": finite_logits,
        "tokens_match": tokens_match,
        "reference_tokens": expected_ids[0, prompt.shape[1]:].tolist(),
        "candidate_tokens": actual_ids[0, prompt.shape[1]:].tolist(),
        "convolution_calls_per_rollout": 4 * steps,
        "baseline_latency_ms": None,
        "candidate_latency_ms": None,
        "baseline_tokens_per_second": None,
        "candidate_tokens_per_second": None,
        "speedup": None,
        "baseline_convolution_fraction": None,
        "profile_note": "Separate instrumented synthetic baseline; not a production bottleneck estimate.",
    }
    if not result["passed"]:
        return result

    # Correctness rollouts also warmed every sequence-length specialization.
    elapsed = {"baseline": [], "candidate": []}
    for repeat in range(repeats):
        order = [("baseline", reference), ("candidate", candidate)]
        if repeat % 2:
            order.reverse()
        for name, convolution in order:
            _synchronize(target)
            started = time.perf_counter()
            _decode(model, prompt, convolution, steps)
            _synchronize(target)
            elapsed[name].append(time.perf_counter() - started)
    baseline = statistics.median(elapsed["baseline"])
    measured = statistics.median(elapsed["candidate"])

    conv_seconds = 0.0

    def profiled_reference(*args):
        nonlocal conv_seconds
        _synchronize(target)
        started = time.perf_counter()
        output = reference(*args)
        _synchronize(target)
        conv_seconds += time.perf_counter() - started
        return output

    _synchronize(target)
    started = time.perf_counter()
    _decode(model, prompt, profiled_reference, steps)
    _synchronize(target)
    profile_seconds = time.perf_counter() - started
    result.update({
        "timing_repeats": repeats,
        "baseline_latency_ms": baseline * 1000,
        "candidate_latency_ms": measured * 1000,
        "baseline_tokens_per_second": steps / baseline,
        "candidate_tokens_per_second": steps / measured,
        "speedup": baseline / measured,
        "baseline_convolution_fraction": conv_seconds / profile_seconds,
        "profile_total_latency_ms": profile_seconds * 1000,
        "profile_convolution_latency_ms": conv_seconds * 1000,
    })
    return result
