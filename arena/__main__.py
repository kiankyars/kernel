"""CLI initializes Triton mode before importing any candidate or evaluator."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["interpret", "cuda"], default="interpret")
    parser.add_argument("--output", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261027, help="Worker mutation seed, not holdout seed")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    for name in ("population", "generations", "cases", "repeats", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.population < 3:
        parser.error("--population must be >=3 (fused, split, and a negative control)")
    if args.population > 64:
        parser.error("--population must be <=64, the size of the valid mutation space")
    if args.cases < 24:
        parser.error("--cases must be >=24 to cover every required dtype and value mode")
    os.environ["TRITON_INTERPRET"] = "1" if args.backend == "interpret" else "0"
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path("artifacts/triton-cache").resolve()))
    import torch
    if args.backend == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA hardware and a CUDA-enabled PyTorch build are required; use --backend interpret here")
    torch.set_num_threads(1)
    from .evolution import run
    report = run(args)
    print(json.dumps({key: report[key] for key in (
        "mode", "evaluated", "correctness_passed", "memory_passed", "validated", "rejected",
        "gpu_winner", "production_eligible", "report_path",
    )}, indent=2))
    return 0 if report["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

