# Choosing the first arena problem

The selected example is **DFlash2 grouped dynamic causal convolution**. Source inspection shows a narrow operation repeated four times per draft decoder layer, implemented as a Python loop over kernel offsets with padding, multiplication, addition, and `addcmul` intermediates. Fusing that loop has a concrete hypothesis: eliminate repeated launches and full-tensor temporaries while preserving the output contract.

This is a provisional choice among important, narrow operations with a visible optimization opportunity. It is **not** evidence that this operation is the largest inference bottleneck, or that its GPU speedup or model-level improvement will be greatest. Attention and dense projections can dominate total inference time. Their shares, and this convolution's share, require representative profiling on the intended hardware.

## Search scope and provenance

Research preceded implementation and compared source, existing optimized kernels, call sites, tests, and benchmarks across these public snapshots:

| Project | Inspected commit | Scope |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm/tree/4bb600439d5eeb5b3fc619c50711f7d700d3671f) | `4bb600439d5eeb5b3fc619c50711f7d700d3671f` | Attention backends, KV writers, residual normalization, norm/quantization and RoPE/cache fusion, sampling, kernel tests, kernel benchmarks, Llama call sites |
| [DFlash](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a) | `07ebd93db9f472af339b644bb70221ad8428328a` | Draft/target generation loop, attention, norms, KV updates, DFlash2 dynamic convolution, candidate selection, sampling/acceptance, model benchmark |

Both were inspected on 2026-10-27. The search is comprehensive across the requested operation families, not an exhaustive survey of every inference system or GPU architecture. No GPU performance was measured during selection.

The user's `/Users/kian/Developer/vllm` and `/Users/kian/Developer/dflash-qrlm` checkouts are not mounted in this cloud machine. A read-only probe of `kiankyars/dflash-qrlm` returned “Repository not found.” That response does not establish that a private repository is absent, and the public DFlash repository is not assumed equivalent to the user's fork.

The following pages could not be read because the environment's network proxy returned CONNECT 403:

- [DFlash paper, arXiv:2602.06036](https://arxiv.org/abs/2602.06036)
- [DFlash project page](https://z-lab.ai/projects/dflash/)
- [DFlash2 article](https://inco.ai/blog/dflash2/)
- [The user-supplied AlphaEvolve article](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

No results from those unread pages are used as evidence. The arena's generate/evaluate/select structure follows the user's recommendation; this document does not independently substantiate claims about AlphaEvolve deployments.

## First filter: importance

The first filter asks whether an operation participates repeatedly in meaningful inference work, rather than whether it admits an attractive isolated benchmark. Per-layer operations qualify directly; per-round operations qualify conditionally when vocabulary width, speculative block size, or host synchronization makes them significant. Actual wall-clock shares remain unknown.

| Operation family | Importance evidence | Importance conclusion |
|---|---|---|
| Attention and paged KV reads | Target verification and draft attention recur each generation round; attention reads scale with context length. DFlash attention has Q/K/V projections, SDPA, and an output projection. [D1], [V1] | Highest-priority production profiling targets; likely dominant in some long-context regimes |
| Dense/quantized projections | Attention projections and MLP execute in each layer. vLLM contains hardware-specialized CUTLASS/Marlin and lower-precision paths. [D1], [V2] | Highest-priority production profiling targets; often substantial compute or weight-memory work |
| **DFlash2 dynamic convolution** | Each draft layer has attention and MLP convolution modules; both call `prepare` and `finish`, totaling four dynamic convolutions per layer. [D2], [D3] | Passes the recurring per-layer filter; limited to DFlash2, with unmeasured model-level share |
| KV-cache writes/layout conversion | Every new key/value must reach its cache; slot mapping, layout, and quantization affect serving. [D1], [V3] | Passes; write cost must be distinguished from larger cache-read/attention costs |
| Residual RMSNorm | Llama invokes normalization before attention and before MLP; DFlash likewise normalizes twice per layer. [D2], [V4] | Passes with broad model coverage; total time share can still be small |
| RoPE and Q/K normalization | Runs in the attention path at every layer, with position-dependent output and cache integration. [D1], [V5] | Passes; fusion opportunity depends on existing backend support |
| Vocabulary filtering/sampling | DFlash's probability path includes softmax, top-p sort, scatter, and sampling; runs per generated/speculative block rather than per layer. [D4], [V6] | Conditional importance, increasing with wide vocabularies and small/fast models |
| DFlash2 candidate selection | Loops serially over draft positions using codebook embeddings, an einsum, and argmax or multinomial. [D5] | Conditional importance; dependency chain and block size matter |
| Speculative scheduling/acceptance | Generation includes host/device synchronization and cache cropping. [D6] | Potentially important, but a reference Python loop is not automatically a production serving bottleneck |

## Second filter: hypothesized improvement and tractability

Only after the importance filter do we compare plausible proportional improvements. The relevant baseline is the actual selected upstream path, not an artificially slow expression chosen to make a result look large. CPU feasibility is an additional implementation constraint, not evidence of production importance.

| Candidate | Visible opportunity | Existing optimization or obstacle | Decision for this example |
|---|---|---|---|
| **DFlash2 grouped dynamic causal convolution** | Replace O(K) eager stages and their temporaries with a fused output-tile kernel; keep accumulators local while reusing causal input values | Low-precision rounding, `addcmul`, strides, and nonfinite boundaries require care; no GPU profile establishes its fraction | **Selected:** strongest concrete fusion opportunity among the inspected narrow per-layer paths |
| Attention / paged KV reads | Better tiling, context partitioning, reuse, or specialized shapes could produce material gains | Current vLLM dispatches to mature FlashAttention, FlashInfer, Triton, and specialized MLA/sparse backends; GPU occupancy and bandwidth cannot be inferred from CPU execution | High-value follow-up after production profiling; too broad for a trustworthy first CPU-only example |
| Dense/quantized GEMM | Algorithmic or precision-specific improvements can affect major inference work | Tensor-core behavior, FP8/FP4 details, hardware-specific layouts, and mature libraries require target-hardware measurement | Defer hardware-dependent search; not rejected for lack of importance |
| KV-cache writing | Fuse layout, quantization, or adjacent producers; avoid extra copies | Existing CUDA writers support multiple layouts and scales; vLLM already fuses RoPE/cache for supported paths | Credible follow-up, but weaker visible marginal opportunity than the selected eager loop |
| Residual RMSNorm | Fuse residual addition, reduction, scaling, and perhaps quantization | Already has vectorized CUDA kernels and RMSNorm/quantization fusions; beating eager Torch does not mean beating vLLM | Strong oracle-development task, weaker evidence of a large new upstream gain |
| RoPE / QK norm fusion | Remove intermediate reads/writes and launches | Existing compiler passes fuse QK norm/RoPE and RoPE/cache; coverage is model/backend dependent | Defer until a documented uncovered regime is identified |
| Sampling/top-p | Avoid full-vocabulary sorting and materialization, exploit small batches | vLLM already uses iterative/split-row Triton and FlashInfer. RNG, ties, logprobs, and statistical equivalence complicate the oracle | Plausibly high upside in specific gaps, but no identified gap outranks the selected concrete loop |
| DFlash2 candidate selector | Fuse repeated small kernels and reuse codebook data | Predecessor token dependencies remain; stochastic correctness and dynamic candidate access expand scope | Promising second DFlash-specific example |
| Scheduling/acceptance | Reduce host round trips and Python dispatch | May be an integration/control-flow task rather than a narrow GPU kernel; serving integrations can already remove reference-loop overhead | Defer until production traces show the relevant path |

The sampling assessment has a useful negative result: vLLM's source explicitly describes a low-batch monolithic top-p path with expensive repeated full-vocabulary sweeps, then provides a split-row remedy [V7]. Finding an expensive pattern is insufficient if the inspected production implementation already remedies it.

For the selected convolution, the hypothesis is deliberately qualitative. Fusing K offsets reduces the number of launches and tensor passes, but it does not remove all input reads, and more register pressure or poor coalescing can erase the benefit. No fixed speedup is promised.

## Selected operation and behavioral contract

The source target is `_grouped_dynamic_convolve` in [D3], used by `GroupedDynamicCausalConv.prepare` and `.finish`. Its inputs are hidden states `hidden[B,L,H]`, dynamic coefficients `dynamic[B,L,K,H/G]`, static coefficients `base[K,H]`, and group size `G`. Dimensions and `G` are positive, `H` must be divisible by `G`, and `1 <= K <= L+1`. Hidden and dynamic coefficients share a dtype; base coefficients use that dtype or fp32 and are converted to the hidden dtype. The grouping maps channel `h` to `floor(h/G)`.

For batch `b`, output position `t`, and channel `h`, let `v_o` be `hidden[b,t-o,h]` when `t >= o`, and padded zero otherwise. Begin with `a = 0`, then visit offsets `o = 0, …, K-1` in that order:

```text
p = cast_to_hidden_dtype(base[o,h]) * v_o
a = a + p
a = addcmul(a, dynamic[b,t,o,floor(h/G)], v_o)
output[b,t,h] = a
```

Dynamic coefficients belong to the **output** position `t`, not the source position `t-o`. The result has the hidden tensor's shape. Inputs must remain unchanged. The PyTorch reference's operation order and dtype conversions are part of the oracle: algebraically combining `(base + dynamic) * value`, changing accumulation precision, or reassociating offsets can change low-precision answers. Nonfinite coefficients multiplying padded zeros also require explicit comparison; a masked load followed by multiplication is not generally equivalent to skipping arithmetic for NaN/Inf.

The upstream implementation uses `view`, so arbitrary strided layouts are not universally accepted there. The arena can test a broader layout contract by comparing the same mathematical operation on copied logical values, but such cases are arena robustness coverage rather than supported upstream layouts. The upstream padding expression also has a valid-shape boundary: offset `L` produces an empty source padded to exactly `L` positions, while an offset greater than `L` produces too many positions. The arena therefore admits `K <= L+1` and rejects larger kernels instead of silently extending the upstream operation.

Kernel-size, shape, dtype, offset, alignment, stride, tail, and special-value variation belong in correctness tests before optimization scores. Production model configuration values must be included when moving to a real DFlash2 checkpoint.

## What constitutes evidence of a useful result

There are separate claims to validate:

1. **Behavior:** an independently expressed PyTorch oracle, unseen randomized cases, mutation checks, and numerical/special-value checks pass before performance is admitted.
2. **Memory safety:** CPU bounds/canary checks or sanitizers exercise relevant host implementations; CUDA memcheck/racecheck on actual hardware remains necessary for GPU promotion. CPU emulation cannot prove GPU race freedom.
3. **Kernel performance:** latency, throughput, and memory are measured on a fixed held-out workload against a named baseline on the same backend and device. Interpreter runtime is not a GPU performance proxy.
4. **Inference impact:** the candidate is substituted into the actual model path, preserving generation behavior, and compared on representative complete inference workloads. A tiny random model or synthetic repeated decoder fixture is an integration test, not a production inference benchmark.

DFlash provides a real-model benchmark with warmup, baseline and speculative generation, TPOT, throughput, and acceptance-length reporting [D7]. Its CUDA synchronization and substantial model checkpoints are not available in this CPU-only machine. Those results must remain explicitly unmeasured here.

If a kernel occupies fraction `f` of baseline inference time and its measured speedup is `s`, the optimistic total speedup is `1 / ((1-f) + f/s)`, before added overhead. This bounds the value of a microbenchmark result. A promotion decision should require actual end-to-end improvement, not merely a favorable kernel ranking.

## Pinned primary sources

- **D1:** [DFlash attention projections, Q/K normalization, RoPE, cache update and SDPA](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L365-L420).
- **D2:** [DFlash decoder layer: prepare/finish around attention and MLP](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L433-L474).
- **D3:** [Grouped dynamic convolution and its module](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L478-L516); [DFlash2 installs two modules per layer](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L635-L651).
- **D4:** [DFlash probability filtering](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L35-L69).
- **D5:** [DFlash2 candidate selector](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L519-L550).
- **D6:** [DFlash generation, verification, acceptance and cache cropping](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L231-L311).
- **D7:** [DFlash real-model benchmark](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/benchmark.py#L230-L284); [supported models and production integrations](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/README.md).
- **V1:** [vLLM attention backend registry](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/vllm/v1/attention/backends/registry.py).
- **V2:** [vLLM quantization kernels](https://github.com/vllm-project/vllm/tree/4bb600439d5eeb5b3fc619c50711f7d700d3671f/csrc/quantization); [RMSNorm/quantization fusion](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/vllm/compilation/passes/fusion/rms_quant_fusion.py).
- **V3:** [vLLM KV layout, slot mapping, NHD/HND and quantization scaling](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/csrc/cache_kernels.cu#L266-L408).
- **V4:** [Llama normalization call sites](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/vllm/model_executor/models/llama.py#L308-L334); [native oracle and tolerances](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/vllm/ir/ops/layernorm.py); [vectorized fused CUDA implementation](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/csrc/layernorm_kernels.cu#L100-L220); [strided and irregular-shape tests](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/tests/kernels/core/test_layernorm.py).
- **V5:** [QK norm/RoPE fusion](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py); [RoPE/cache fusion](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/vllm/compilation/passes/fusion/rope_kvcache_fusion.py).
- **V6:** [vLLM sampler implementations and dispatch](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/vllm/v1/sample/ops/topk_topp_sampler.py#L504-L676).
- **V7:** [Existing low-batch top-p split-row remedy](https://github.com/vllm-project/vllm/blob/4bb600439d5eeb5b3fc619c50711f7d700d3671f/vllm/v1/sample/ops/topk_topp_triton.py#L965-L1005).
