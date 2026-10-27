"""Separate symbolic bounds checks from independent numerical correctness.

TileLens's SymbolicSanitizer._op_store_overrider checks pointer/mask expressions
without storing values (tilelens/clients/sanitizer/sanitizer.py). Consequently a
symbolic run must never compare its output values to the numerical oracle.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Triton's interpreter resolves its language from the kernel's module globals.
import triton
import triton.language as tl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=481516)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--probe-oob", action="store_true",
                        help="intentionally invalid store; sanitizer must reject it")
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("--cases must be positive")
    if not args.probe_oob and args.spec is None:
        parser.error("--spec is required unless --probe-oob is used")
    import torch
    from arena.oracle import evaluate, make_cases, _guarded_tensor
    from arena.spec import CandidateSpec
    torch.set_num_threads(1)

    if args.device == "cpu":
        from tilelens.core.config import config
        if not config.cli_active or not config.enable_sanitizer:
            raise RuntimeError("CPU memory certification requires active tile-sanitizer with ENABLE_SANITIZER=1")
        from tilelens.core.trace import launches
        from tilelens.clients.sanitizer.sanitizer import SymbolicSanitizer
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA memory checks require an available CUDA device")

    if args.probe_oob:
        @triton.jit
        def invalid_store(Out):
            tl.store(Out + 1, 123.0)

        # This is deliberately outside a one-element allocation. CPU invocation
        # is refused above unless the symbolic sanitizer is actively installed.
        out = torch.empty(1, device=args.device)
        print("ARENA_OOB_PROBE=armed", flush=True)
        invalid_store[(1,)](out)
        if args.device == "cuda":
            torch.cuda.synchronize()
        print("ARENA_OOB_PROBE=undetected", flush=True)
        return 1

    spec = CandidateSpec(**json.loads(args.spec))
    cases = make_cases(args.seed, args.cases, args.device)
    if args.device == "cpu":
        import random
        from arena import kernels
        traced_kernel = kernels._split_step if spec.variant == "split" else kernels._fused
        manager = getattr(traced_kernel, "client_manager", None)
        client = manager.get_client("sanitizer") if manager is not None else None
        if not isinstance(client, SymbolicSanitizer) or not client.abort_on_error:
            raise RuntimeError("candidate is not wrapped in an aborting symbolic sanitizer")
        before = len(launches)
        cases_checked = 0
        for index, case in enumerate(cases):
            out = _guarded_tensor(case.hidden.shape, case.hidden.dtype,
                                  torch.Generator().manual_seed(index), random.Random(index),
                                  case.hidden.device, strided=True)
            before_case = len(launches)
            kernels.convolve(case.hidden, case.dynamic, case.base, case.group_size,
                             spec=spec, out=out)
            if len(launches) <= before_case:
                raise RuntimeError("sanitizer recorded no kernel launch for this case")
            cases_checked += 1
        result = {"passed": cases_checked == args.cases,
                  "memory_cases_checked": cases_checked,
                  "numerical_cases_checked": 0,
                  "sanitizer_launches": len(launches) - before,
                  "sanitizer": "tilelens-symbolic", "sanitizer_active": True}
    else:
        result = evaluate(spec, cases)
        result.update(memory_cases_checked=result["cases_run"],
                      numerical_cases_checked=result["cases_run"],
                      sanitizer="compute-sanitizer")
    print("ARENA_MEMORY_RESULT=" + json.dumps(result), flush=True)
    return 0 if result["passed"] and result["memory_cases_checked"] == args.cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
