"""Bounded worker mutations, frozen submissions, hidden tests, durable results.

Workers receive typed specs and mutation seeds only. They do not execute model-
written Python. Local files/subprocesses are NOT a sandbox for hostile programs.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import itertools
import multiprocessing
from pathlib import Path
import platform
import random
import secrets
import sqlite3
import time

from .spec import CandidateSpec


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def mutate(payload):
    """The entire worker capability: produce one bounded declarative candidate."""
    parents, seed = payload
    rng = random.Random(seed)
    parent = rng.choice(parents) if parents else asdict(CandidateSpec())
    choices = {"block_size": (32, 64, 128, 256), "num_warps": (1, 2, 4, 8),
               "unroll": (True, False), "variant": ("fused", "split")}
    key = rng.choice(list(choices))
    spec = dict(parent)
    spec[key] = rng.choice([value for value in choices[key] if value != spec[key]])
    return asdict(CandidateSpec(**spec))


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(args):
    import torch
    import triton
    from .benchmark import microbenchmark
    from .inference import integration_check
    from .kernels import source_digest
    from .oracle import evaluate, make_cases
    from .safety import check_memory

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.backend == "interpret" else "cuda"
    root = Path(__file__).resolve().parent
    source_files = {p.name: p.read_text() for p in sorted(root.glob("*.py"))}
    source_files["check_memory.py"] = (root.parent / "scripts/check_memory.py").read_text()
    snapshot = output / "source"
    snapshot.mkdir()
    for name, content in source_files.items():
        (snapshot / name).write_text(content)
    run_info = {
        "backend": args.backend, "device": device, "python": platform.python_version(),
        "torch": torch.__version__, "triton": triton.__version__,
        "cpu": platform.machine(), "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "source_sha256": digest(source_files), "worker_seed": args.seed,
        "population": args.population, "generations": args.generations,
        "cases_per_candidate": args.cases, "timing_repeats": args.repeats,
        "timestamp_unix": time.time(),
        "oracle_scope": "DFlash2 reference numerical contract; not distributional model quality",
    }
    _dump(output / "run.json", run_info)
    database = sqlite3.connect(output / "arena.sqlite")
    database.execute("CREATE TABLE evaluations (candidate_id TEXT PRIMARY KEY, generation INTEGER, source_hash TEXT, spec_json TEXT, result_json TEXT)")
    rows = []
    parents = []
    private_seeds = {}
    rng = random.Random(args.seed)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for generation in range(args.generations):
                initial = [asdict(CandidateSpec(variant=variant)) for variant in ("fused", "split", "broken_shift")] if generation == 0 else []
                payloads = [(parents, rng.getrandbits(63)) for _ in range(args.population - len(initial))]
                specs = initial + list(pool.map(mutate, payloads))
                # Remove accidental duplicates without consulting hidden outcomes.
                seen = set()
                for index, spec in enumerate(specs):
                    attempts = 0
                    while digest(spec) in seen and attempts < 256:
                        spec = mutate((parents, rng.getrandbits(63)))
                        attempts += 1
                    if digest(spec) in seen:
                        remaining = [asdict(CandidateSpec(block, warps, unroll, variant))
                                     for block, warps, unroll, variant in itertools.product(
                                         (32, 64, 128, 256), (1, 2, 4, 8), (True, False), ("fused", "split"))]
                        spec = next(item for item in remaining if digest(item) not in seen)
                    specs[index] = spec
                    seen.add(digest(spec))
                submissions = [{"spec": spec, "source_hash": source_digest(CandidateSpec(**spec))} for spec in specs]
                commit = {"generation": generation, "submissions": submissions,
                          "parent_specs": parents, "evaluator_sha256": run_info["source_sha256"]}
                # Freeze and write the candidate roster BEFORE creating holdouts.
                commit_path = output / f"generation-{generation}-submissions.json"
                _dump(commit_path, commit)
                commit_path.chmod(0o444)
                seed = secrets.randbits(63)
                private_seeds[str(generation)] = seed
                secret_path = output / "replay-seeds.private.json"
                # Withheld from mutation workers, not a security claim about the OS.
                fd = secret_path.open("w")
                secret_path.chmod(0o600)
                with fd:
                    json.dump(private_seeds, fd)
                _dump(output / f"generation-{generation}-seal.json", {
                    "submission_sha256": digest(commit), "seed_commitment": digest(seed),
                    "correctness_cases": args.cases, "performance_seed_commitment": digest(seed ^ 0xC0FFEE),
                })
                cases = make_cases(seed, args.cases, device)
                generation_rows = []
                for index, submission in enumerate(submissions):
                    spec = CandidateSpec(**submission["spec"])
                    candidate_id = f"g{generation}-{index}-{submission['source_hash'][:12]}"
                    candidate_dir = output / candidate_id
                    candidate_dir.mkdir()
                    print(f"Evaluating {candidate_id}: {spec.variant}, block={spec.block_size}, warps={spec.num_warps}", flush=True)
                    row = {"candidate_id": candidate_id, "generation": generation, **submission,
                           "correctness": evaluate(spec, cases), "memory": None,
                           "integration": None, "microbenchmark": None, "gpu_fitness": None,
                           "production_eligible": False, "status": "correctness_rejected"}
                    if row["correctness"]["passed"]:
                        row["memory"] = check_memory(submission["spec"], candidate_dir, device=device,
                                                     seed=seed ^ 0xB0A0D5, cases=6)
                        row["status"] = "memory_rejected"
                        if row["memory"]["passed"]:
                            # The integration function checks all logits/tokens
                            # before it measures either implementation.
                            try:
                                row["integration"] = integration_check(spec, seed=seed % (2**31), device=device, repeats=args.repeats)
                                row["status"] = "integration_rejected"
                                if row["integration"]["passed"]:
                                    row["microbenchmark"] = microbenchmark(spec, seed ^ 0xC0FFEE, device=device, repeats=args.repeats)
                                    row["status"] = "cpu_validated" if device == "cpu" else "gpu_provisional"
                                    if device == "cuda":
                                        # Both micro and complete synthetic decode
                                        # must improve; a micro-only win cannot win.
                                        row["gpu_fitness"] = min(row["microbenchmark"]["geometric_speedup"], row["integration"]["speedup"])
                            except Exception as exc:
                                row["status"] = "evaluation_error"
                                row["error"] = f"{type(exc).__name__}: {exc}"
                    _dump(candidate_dir / "result.json", row)
                    database.execute("INSERT INTO evaluations VALUES (?, ?, ?, ?, ?)",
                                     (candidate_id, generation, submission["source_hash"], json.dumps(submission["spec"]), json.dumps(row)))
                    database.commit()
                    rows.append(row)
                    generation_rows.append(row)
                    print(f"  {row['status']}", flush=True)
                eligible = [r for r in generation_rows if r["status"] in ("cpu_validated", "gpu_provisional")]
                if device == "cuda":
                    eligible.sort(key=lambda row: row["gpu_fitness"], reverse=True)
                else:
                    # Interpreter time is never a GPU optimization objective.
                    rng.shuffle(eligible)
                parents = [r["spec"] for r in eligible[:max(1, args.population // 2)]]
                if not parents:
                    break
    finally:
        database.close()
    # An evaluator edited during the run invalidates provenance and the result.
    current = {p.name: p.read_text() for p in sorted(root.glob("*.py"))}
    current["check_memory.py"] = (root.parent / "scripts/check_memory.py").read_text()
    if digest(current) != run_info["source_sha256"]:
        raise RuntimeError("Evaluator or candidate source changed during run; rerun from a frozen checkout")
    correct = sum(r["correctness"]["passed"] for r in rows)
    safe = sum(bool(r["memory"] and r["memory"]["passed"]) for r in rows)
    viable = [r for r in rows if r["status"] in ("cpu_validated", "gpu_provisional")]
    winners = [r for r in viable if r["gpu_fitness"] is not None and r["gpu_fitness"] > 1.0]
    winner = max(winners, key=lambda r: r["gpu_fitness"])["candidate_id"] if winners else None
    report = {
        "mode": "CPU correctness exploration; GPU fitness unavailable" if device == "cpu" else "GPU evolution with provisional synthetic integration fitness",
        "evaluated": len(rows), "correctness_passed": correct, "memory_passed": safe, "validated": len(viable),
        "rejected": len(rows) - len(viable), "gpu_winner": winner,
        "production_eligible": False,
        "production_gate": "Run scripts/benchmark_dflash.py on pretrained local models; no production claim from synthetic inference.",
        "report_path": str(output / "report.json"), "run": run_info, "candidates": rows,
        "scope": "Template-generated Triton programs with bounded mutation; not arbitrary LLM-generated code or a hostile-code sandbox.",
    }
    _dump(output / "report.json", report)
    return report
