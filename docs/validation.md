# Validation on the CPU cloud machine

Validated on 2026-10-27 with Python 3.12.14, CPU PyTorch 2.9.1, Triton 3.6.0,
TileLens 3.1.1, and NumPy 2.2.6. The machine has no CUDA device. Setup ran twice
successfully using the same dependency constraints.

| Check | Result | Meaning |
|---|---|---|
| Full pytest suite | **92 passed**, 0 failed | Numerical/storage contracts, invalid inputs, actual kernels, autoregressive integration, scoped adapter, controller gates and memory-tool evidence |
| Evolution run | **8 evaluations**, 2 generations | Actual worker-generated specifications, source commitments, randomized holdouts, durable SQLite history |
| Correct candidates | **7 passed × 24 cases = 168** | FP32/FP16/BF16, shapes, strides, offsets, boundaries, NaN/Inf and mutation guards |
| Incorrect negative control | **Rejected on its first case** | No memory, integration or performance result awarded |
| Symbolic memory checks | **42 cases, 82 traced launches** | All seven viable candidates passed TileLens bounds checks |
| Memory-tool negative control | **7 expected OOB rejections** | Certification requires the active tool to reject an intentional invalid store |
| Autoregressive integration | **7 complete rollouts passed** | Four generated tokens and four full logit tensors checked per candidate; exact token matches |
| Maximum integration logit error | **9.54e-7** | Below the declared FP32 tolerance on this synthetic model |
| CUDA offline compilation | **6 passed** | Fused/split × FP32/FP16/BF16 compile to sm80 PTX/cubin without a driver |
| Native GPU / pretrained DFlash | **Not run** | Requires CUDA hardware, compatible CUDA dependencies and local pretrained checkpoints |

The four pytest warnings are expected invalid-arithmetic warnings from deliberate
Inf/NaN inputs. Those cases passed exact nonfinite-location checks; they were
neither skipped nor suppressed to make tests pass. Unit tests that simulate GPU
controller behavior are identified as mocked tests and do not constitute GPU
execution.

The BF16 oracle exposed an interpreter cast discrepancy. The kernel now performs
explicit round-to-nearest-even at the reference's rounding boundaries; tolerances
were not loosened. TileLens symbolic stores do not write numerical values, so
its evidence is kept separate from the independent numerical oracle. Both facts
are regression-tested.

CPU interpreter timing is intentionally uncompetitive with optimized PyTorch CPU
operators: the default fused candidate's complete four-token synthetic rollout
took about **251 ms**, versus **1.26 ms** for the reference. This is simulator
overhead, **not a GPU slowdown or speedup measurement**. No candidate was selected
as a GPU winner. The synthetic profile's convolution fraction is also not a
production bottleneck estimate.

Commands used:

```bash
bash scripts/setup.sh
TRITON_INTERPRET=1 /workspace/.venvs/kernel-arena/bin/python -m pytest -q
/workspace/.venvs/kernel-arena/bin/python scripts/check_codegen.py --output artifacts/codegen.json
/workspace/.venvs/kernel-arena/bin/python -m arena \
  --output artifacts/validated-run --population 4 --generations 2 --cases 24 --repeats 3
```

The checked-in [example evidence](example-results.json) contains source/spec
hashes, per-candidate outcomes and offline compilation hashes. The local full
[run report](../artifacts/validated-run/report.json), SQLite database, sealed
submission manifests, source snapshot, raw timing samples and sanitizer logs
remain in `artifacts/validated-run/` (generated, Git-ignored). A source-hash check
confirmed the evaluator and kernel did not change during that run.

Remaining promotion gates are actual native GPU correctness and memcheck/racecheck,
sealed hardware benchmarks, and the supplied real pretrained DFlash runner on
representative prompts. All must refer to the same candidate. No production
speedup, GPU winner, model-quality result or successful environment publication
is claimed by this report.
