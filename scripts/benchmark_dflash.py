#!/usr/bin/env python3
"""Real pretrained DFlash2 benchmark; CUDA and local checkpoints are required.

Prompt JSON: {"prompts": [[1, 42, 17], [1, 9, 31]], "max_new_tokens": 64}.
Candidate JSON is a plain CandidateSpec object. No tokenization, downloads,
remote model code, or synthetic-model fallback is performed.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from arena.dflash_adapter import DFLASH_COMMIT, patched_convolution, validate_upstream
from arena.spec import CandidateSpec


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("Real DFlash2 inference benchmark requires CUDA hardware; CPU interpreter results cannot replace it")
    if os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("Unset TRITON_INTERPRET for real CUDA inference benchmarks")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(directory: Path) -> dict:
    directory = directory.expanduser().resolve(strict=True)
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError(f"local checkpoint directory must contain config.json: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.is_file()
                   and (p.suffix in (".json", ".safetensors")))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"local checkpoint requires safetensors weights: {directory}")
    return {
        "directory": str(directory),
        "files": [{"path": str(p.relative_to(directory)), "size": p.stat().st_size,
                   "mtime_ns": p.stat().st_mtime_ns, "sha256": _sha256(p)} for p in files],
    }


def _unchanged(manifest: dict):
    root = Path(manifest["directory"])
    current = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                     if p.is_file() and p.suffix in (".json", ".safetensors"))
    if current != [item["path"] for item in manifest["files"]]:
        raise RuntimeError("checkpoint file membership changed during benchmark")
    for item in manifest["files"]:
        stat = (root / item["path"]).stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise RuntimeError("checkpoint changed during benchmark")


def read_workload(path: Path) -> tuple[list[list[int]], int]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict) or set(obj) != {"prompts", "max_new_tokens"}:
        raise ValueError("prompt JSON must contain exactly prompts and max_new_tokens")
    prompts, steps = obj["prompts"], obj["max_new_tokens"]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a nonempty list of token-id lists")
    for prompt in prompts:
        if (not isinstance(prompt, list) or not prompt
                or any(type(token) is not int or not 0 <= token < 2**31 for token in prompt)):
            raise ValueError("each prompt must contain nonnegative integer token IDs")
    if type(steps) is not int or steps < 2:
        raise ValueError("max_new_tokens must be an integer >= 2 to exercise draft inference")
    return prompts, steps


def _same_generation(left: dict, right: dict):
    if left["output_ids"] != right["output_ids"]:
        raise RuntimeError("correctness gate failed: generated token IDs differ; no timing result is accepted")
    if left["acceptance_lengths"] != right["acceptance_lengths"]:
        raise RuntimeError("correctness gate failed: speculative acceptance lengths differ; no timing result is accepted")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


@torch.inference_mode()
def run_benchmark(args) -> dict:
    require_cuda()  # fail before imports, checkpoint reads, or model loading
    if args.warmups < 1 or args.repeats < 3:
        raise ValueError("use at least one warmup and three paired measured repeats")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError("CUDA device index is unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    prompts, max_new_tokens = read_workload(args.prompts)
    prompt_hash = _sha256(args.prompts)
    candidate_json = args.candidate.read_bytes()
    spec = CandidateSpec(**json.loads(candidate_json))
    if spec.variant == "broken_shift":
        raise ValueError("the deliberately broken negative-control kernel is not a production candidate")

    # Local paths plus the explicit transformers flags below prohibit model
    # downloads and dynamic remote code. Safetensors avoids pickle weights.
    target_manifest = checkpoint_manifest(args.target)
    draft_manifest = checkpoint_manifest(args.draft)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import dflash.model as module
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install the pinned public DFlash checkout and its compatible transformers package first") from exc
    operation_hash = validate_upstream(module)
    from arena.kernels import source_digest

    candidate_hash = source_digest(spec)
    module_hash = _sha256(Path(module.__file__))
    dtype = getattr(torch, args.dtype)
    kwargs = dict(local_files_only=True, trust_remote_code=False,
                  use_safetensors=True, attn_implementation="sdpa", dtype=dtype)
    target = AutoModelForCausalLM.from_pretrained(target_manifest["directory"], **kwargs).to(device).eval()
    draft = module.DFlash2DraftModel.from_pretrained(draft_manifest["directory"], **kwargs).to(device).eval()
    target.requires_grad_(False)
    draft.requires_grad_(False)
    if draft.block_size <= 1 or not any(getattr(layer, "attention_conv", None) is not None for layer in draft.layers):
        raise ValueError("checkpoint must be a DFlash2 model with active convolution and block_size > 1")
    vocab = target.get_input_embeddings().num_embeddings
    if any(max(prompt) >= vocab for prompt in prompts):
        raise ValueError("prompt token ID lies outside target vocabulary")
    inputs = [torch.tensor([prompt], dtype=torch.long, device=device) for prompt in prompts]
    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "device": str(device), "gpu_name": properties.name,
        "gpu_uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": properties.multi_processor_count,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "triton": _version("triton"), "transformers": _version("transformers"),
        "dflash": _version("dflash"), "python": platform.python_version(),
    }
    hardware_hash = hashlib.sha256(json.dumps(hardware, sort_keys=True).encode()).hexdigest()
    workload = {
        "target": target_manifest, "draft": draft_manifest,
        "prompt_file_sha256": prompt_hash, "prompts": prompts,
        "max_new_tokens": max_new_tokens, "temperature": 0.0, "stop_token_ids": None,
        "dtype": args.dtype, "block_size": draft.block_size,
        "dflash_module_sha256": module_hash, "operation_ast_sha256": operation_hash,
    }
    workload_hash = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()

    def trial(index, candidate, measured=False):
        context = patched_convolution(module, spec) if candidate else nullcontext(None)
        with context as calls:
            torch.cuda.synchronize(device)
            initial_allocated = torch.cuda.memory_allocated(device)
            if measured:
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            stats = module.dflash_generate(
                draft, target, inputs[index], max_new_tokens=max_new_tokens,
                stop_token_ids=None, temperature=0.0, return_stats=True,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) if measured else None
            if candidate and calls["calls"] == 0:
                raise RuntimeError("candidate was never called by the real DFlash2 model")
        result = {
            "output_ids": stats.output_ids.cpu().tolist(),
            "acceptance_lengths": [int(x) for x in stats.acceptance_lengths],
            "output_tokens": int(stats.num_output_tokens),
            "candidate_calls": calls["calls"] if calls else 0,
        }
        if measured:
            result.update({
                "latency_s": elapsed,
                "tokens_per_second": stats.num_output_tokens / elapsed,
                "upstream_tpot_s": float(stats.time_per_output_token),
                "time_to_first_token_s": float(stats.time_to_first_token),
                "peak_allocated_bytes": peak,
                "incremental_peak_allocated_bytes": max(0, peak - initial_allocated),
            })
        return result

    # Validate every prompt before starting any measured pair. Warmups also
    # check equality; every measured sample is rechecked after its timer ends.
    expected = []
    for index in range(len(inputs)):
        baseline = trial(index, False)
        candidate = trial(index, True)
        _same_generation(baseline, candidate)
        expected.append(baseline)
    for _ in range(args.warmups):
        for index in range(len(inputs)):
            for candidate in (False, True):
                _same_generation(expected[index], trial(index, candidate))
    samples = []
    for repeat in range(args.repeats):
        pair = {"repeat": repeat, "baseline": [], "candidate": []}
        # Alternate pair order to reduce systematic thermal/order bias.
        for candidate in ((False, True) if repeat % 2 == 0 else (True, False)):
            name = "candidate" if candidate else "baseline"
            for index in range(len(inputs)):
                result = trial(index, candidate, measured=True)
                _same_generation(expected[index], result)
                pair[name].append(result)
        samples.append(pair)
    _unchanged(target_manifest)
    _unchanged(draft_manifest)
    if (_sha256(args.prompts) != prompt_hash or args.candidate.read_bytes() != candidate_json
            or source_digest(spec) != candidate_hash or _sha256(Path(module.__file__)) != module_hash):
        raise RuntimeError("candidate, workload, or DFlash source changed during the benchmark")

    summary = {}
    for name in ("baseline", "candidate"):
        suite_seconds = [sum(row["latency_s"] for row in pair[name]) for pair in samples]
        suite_throughput = [sum(row["output_tokens"] for row in pair[name]) / seconds
                            for pair, seconds in zip(samples, suite_seconds)]
        summary[name] = {
            "median_suite_latency_s": statistics.median(suite_seconds),
            "median_suite_tokens_per_second": statistics.median(suite_throughput),
            "median_upstream_tpot_s": statistics.median(row["upstream_tpot_s"] for pair in samples for row in pair[name]),
            "max_peak_allocated_bytes": max(row["peak_allocated_bytes"] for pair in samples for row in pair[name]),
            "max_incremental_peak_allocated_bytes": max(row["incremental_peak_allocated_bytes"] for pair in samples for row in pair[name]),
        }
    summary["median_paired_suite_speedup"] = statistics.median(
        sum(row["latency_s"] for row in pair["baseline"])
        / sum(row["latency_s"] for row in pair["candidate"]) for pair in samples
    )
    return {
        "kind": "real_pretrained_dflash2_cuda_inference", "correctness_passed": True,
        "production_inference_verified": True, "automatic_promotion": False,
        "gpu_sanitizer_verified": False,
        "limitations": ["CUDA sanitizer evidence must be supplied separately before promotion",
                        "Results apply only to these checkpoints, prompts, decoding policy and hardware",
                        "TPOT is upstream total_decode_time / num_output_tokens; throughput includes prefill"],
        "dflash_reference_commit": DFLASH_COMMIT, "candidate": asdict(spec),
        "candidate_source_sha256": candidate_hash,
        "candidate_spec_file_sha256": hashlib.sha256(candidate_json).hexdigest(),
        "workload_sha256": workload_hash, "workload": workload,
        "hardware_sha256": hardware_hash, "hardware": hardware,
        "warmups": args.warmups, "repeats": args.repeats,
        "summary": summary, "samples": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="local target model directory with safetensors")
    parser.add_argument("--draft", type=Path, required=True, help="local pretrained DFlash2 draft directory")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite an earlier benchmark: {args.output}")
        report = run_benchmark(args)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        # Never leave an earlier result at this path looking like this run.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(serialized)
    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        print(f"DFlash benchmark unavailable or rejected: {exc}", file=sys.stderr)
        return 1
    print(f"Saved real-model benchmark: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
