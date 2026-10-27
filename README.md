# A GPU-kernel evolution arena that can be validated on a CPU

This example evolves **DFlash2 grouped dynamic causal convolution**. It generates
actual Triton kernels from bounded worker mutations, checks them against a
trusted PyTorch reference, runs memory checks, records results in SQLite, and
tests complete autoregressive inference before accepting performance results.

The target was selected after comparing attention, matrix multiplication,
KV-cache operations, RMSNorm, RoPE, quantization, sampling, and DFlash scheduling.
See [the importance-first research and pinned sources](docs/research.md).
The selected operation runs four times per DFlash2 draft layer and its reference
repeatedly materializes padding and intermediates. Fusing it is a plausible
high-upside experiment. **Its production bottleneck share and GPU speedup have
not been measured.**

## Run here, without CUDA

Python 3.12, `uv`, and a Linux x86-64 host are expected. The setup script installs
CPU PyTorch 2.9.1, Triton 3.6.0, TileLens 3.1.1, and pytest in a separate virtual
environment. It does not install CUDA drivers or a virtual GPU.

```bash
cd /workspace/kernel
bash scripts/setup.sh
source /workspace/.venvs/kernel-arena/bin/activate
TRITON_INTERPRET=1 python -m pytest -q
python -m arena --output artifacts/my-run --population 4 --generations 2 --cases 24
```

Use a new output directory for each run; existing results are never overwritten.
Triton's interpreter executes GPU-program semantics on CPU. Its wall time is
**not an estimate of GPU latency**. CPU mode explores correctness and structural
diversity; it deliberately assigns no GPU fitness or winner.

Each run saves `report.json`, `arena.sqlite`, the evaluated source snapshot,
candidate specifications and hashes, submission commitments, per-candidate
sanitizer logs, numerical results, and separately labeled CPU timing diagnostics.
`replay-seeds.private.json` records withheld evaluation seeds for auditing;
workers receive only parent specifications and mutation seeds. These are private
from the bounded worker interface, not secure against arbitrary local code.

## What evolves and what is checked

Workers mutate block size, warp count, static versus dynamic offset unrolling,
and fused versus one-launch-per-offset implementations. This is a working,
finite template search, not an LLM service or arbitrary-code sandbox. A
deliberately wrong causal-index mutation verifies rejection. GPU mode selects
parents by the minimum of kernel speedup and complete synthetic-inference
speedup; CPU mode keeps diverse correct candidates without performance ranking.

The operation consumes `hidden[B,L,H]`, `dynamic[B,L,K,H/group_size]`, and
`base[K,H]`. The domain is nonempty tensors, `1 <= K <= L+1`, and a positive
group size dividing `H`. FP32, FP16, and BF16 are supported; dynamic matches
hidden's dtype, while base may also be FP32. Inputs may have positive or
broadcast strides and storage offsets; output must not overlap itself or inputs.
This is inference-only and does not provide autograd.

The trusted reference preserves the upstream arithmetic order and intermediate
rounding. Tests include randomized shapes, strides, offsets, dtypes, seeds,
extreme values, NaNs, infinities, singleton dimensions, tails and causal padding.
Finite tolerances are explicit in `arena/oracle.py`; nonfinite locations and
signs must match exactly. Whole input storage must remain unchanged and output
storage guards must survive. Numerical correctness is an absolute gate before
candidate timing. At least 24 cases cover the full built-in value/dtype schedule.

TileLens separately checks symbolic memory bounds; it does not compute numerical
outputs. Sentinel checks supplement it. Neither tool establishes native GPU
race safety. The tiny inference model has causal attention, an MLP, residuals,
and four dynamic convolutions per layer; every generated token and all logits
are checked. It uses random weights and no KV cache, so its timings and measured
convolution fraction describe only that synthetic workload.

## Move the experiment to a GPU

Use a separate environment with CUDA-enabled PyTorch, Triton 3.6.0 and NVIDIA
Compute Sanitizer. Do **not** run the CPU setup script over that environment.
Install this package with `python -m pip install --no-deps -e .` after installing
compatible dependencies. Then run:

```bash
TRITON_INTERPRET=0 python -m arena --backend cuda --output artifacts/gpu-run
```

CUDA mode requires native correctness, `compute-sanitizer` memcheck/racecheck,
and sealed microbenchmarks with synchronized host-inclusive latency, throughput,
and incremental peak CUDA allocation. A candidate that loses the synthetic
end-to-end benchmark cannot win on its microbenchmark alone. GPU results remain
provisional until actual pretrained inference passes.

`scripts/check_codegen.py` can compile both templates and all three dtypes to
sm80 CUDA code without a device. Compilation is not evidence of GPU execution.

For real DFlash2 inference, install the public DFlash revision recorded in
`docs/research.md` and its compatible model dependencies in your GPU environment.
Provide local safetensors checkpoints and a fixed, separately withheld workload:

```json
{"prompts": [[1, 42, 17], [1, 9, 31]], "max_new_tokens": 64}
```

Use real tokenized evaluation prompts for meaningful results. Save one candidate
specification, for example:

```json
{"block_size": 128, "num_warps": 4, "unroll": true, "variant": "fused"}
```

```bash
TRITON_INTERPRET=0 python scripts/benchmark_dflash.py \
  --target /models/target --draft /models/dflash2 \
  --prompts /evaluation/prompts.json --candidate /evaluation/candidate.json \
  --output artifacts/dflash-result.json
```

The runner verifies the upstream operation's syntax against its pinned source,
hashes local checkpoint/workload files, patches only the convolution within a
restoring context, checks generated IDs and speculative acceptance lengths,
then measures complete inference in alternating baseline/candidate order.
It requires real candidate calls, uses local files only, and never silently
falls back to the tiny model. This GPU/checkpoint-dependent path is supplied but
has not been executed on this CPU host. A meaningful upstream claim requires
that result plus correctness/sanitizer evidence on the **same candidate and
hardware**, representative workloads, and repeatable improvements beyond noise.

See [validation results](docs/validation.md) for the evidence from this machine.
