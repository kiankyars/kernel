#!/usr/bin/env python3
"""Compile current Triton candidates for sm80 without a CUDA driver or GPU.

This checks native code generation, not execution, memory safety or performance.
Run in a separate process from the interpreter, for example::

    TRITON_INTERPRET=0 python scripts/check_codegen.py --output artifacts/codegen.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def main() -> int:
    # Re-exec before importing Triton if inherited interpreter mode was enabled.
    # A fresh process avoids decorator objects created under the other backend.
    if os.environ.get("TRITON_INTERPRET") != "0":
        environment = dict(os.environ, TRITON_INTERPRET="0")
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON evidence here as well as stdout")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "kernel-arena-codegen-cache")
    args = parser.parse_args()
    os.environ["TRITON_CACHE_DIR"] = str(args.cache_dir.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from arena.kernels import _fused, _split_step, source_digest
    from arena.spec import CandidateSpec

    constants = {
        "B": 1, "L": 3, "H": 7, "GROUP": 1,
        "HB": 21, "HL": 7, "HH": 1,
        "DB": 84, "DL": 28, "DK": 7, "DG": 1,
        "BK": 7, "BH": 1, "OB": 21, "OL": 7, "OH": 1, "BLOCK": 32,
    }
    results = []
    for dtype in ("fp32", "fp16", "bf16"):
        for variant, function in (("fused", _fused), ("split", _split_step)):
            spec = CandidateSpec(block_size=32, num_warps=1, unroll=True, variant=variant)
            constexprs = dict(constants)
            constexprs.update(
                {"K": 4, "UNROLL": True, "BROKEN": False}
                if variant == "fused" else {"OFFSET": 3}
            )
            signature = {
                "Hidden": f"*{dtype}", "Dynamic": f"*{dtype}",
                "Base": "*fp32", "Out": f"*{dtype}",
            }
            entry = {"variant": variant, "dtype": dtype, "base_dtype": "fp32",
                     "candidate_source_sha256": source_digest(spec)}
            try:
                source = ASTSource(function, signature=signature, constexprs=constexprs)
                compiled = triton.compile(
                    source, target=GPUTarget("cuda", 80, 32),
                    options={"num_warps": 1, "enable_fp_fusion": False},
                )
                entry.update(
                    status="passed",
                    ptx_sha256=hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest(),
                    cubin_sha256=hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
                )
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            results.append(entry)
    passed = all(result["status"] == "passed" for result in results)
    report = {
        "status": "passed" if passed else "failed",
        "purpose": "offline CUDA compilation only",
        "target": "sm80", "cuda_runtime_executed": False,
        "gpu_latency_ms": None, "torch_version": torch.__version__,
        "triton_version": triton.__version__, "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
