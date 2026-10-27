"""Sealed-workload diagnostics; interpreter timing is never GPU fitness."""
from dataclasses import asdict
import math
import random
import statistics
import time

import torch

from .kernels import convolve
from .oracle import assert_equivalent, reference


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def microbenchmark(spec, seed, *, device="cpu", repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    cuda = torch.device(device).type == "cuda"
    # CPU fixtures deliberately small: interpreter overhead is diagnostic only.
    shapes = [(1, 8, 128, 3, 16), (2, 13, 256, 4, 32)] if not cuda else [
        (1, 4, 4096, 3, 32), (1, 16, 4096, 5, 32),
        (2, 32, 8192, 5, 64), (4, 64, 4096, 3, 32),
    ]
    generator = torch.Generator().manual_seed(seed % (2**63))
    order = random.Random(seed)
    rows = []
    for batch, length, width, offsets, group in shapes:
        args = tuple(torch.randn(shape, generator=generator).to(device) for shape in (
            (batch, length, width), (batch, length, offsets, width // group), (offsets, width),
        ))
        calls = {"reference": lambda: reference(*args, group),
                 "candidate": lambda: convolve(*args, group, spec=spec)}
        # These shapes are additional withheld correctness cases. Never time a
        # candidate on a new shape until it has passed on precisely that input.
        assert_equivalent(calls["candidate"](), calls["reference"]())
        for call in calls.values():
            for _ in range(2):
                call()
        times = {name: [] for name in calls}
        peaks = {name: None for name in calls}
        for _ in range(repeats):
            names = list(calls)
            order.shuffle(names)
            for name in names:
                synchronize(device)
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                    initial = torch.cuda.memory_allocated(device)
                start = time.perf_counter_ns()
                output = calls[name]()
                synchronize(device)
                times[name].append((time.perf_counter_ns() - start) / 1e6)
                if cuda:
                    peak = max(0, torch.cuda.max_memory_allocated(device) - initial)
                    peaks[name] = max(peaks[name] or 0, peak)
                del output
        median = {name: statistics.median(values) for name, values in times.items()}
        rows.append({
            "shape": [batch, length, width], "kernel_size": offsets, "group_size": group,
            "dtype": "float32", "latency_ms": median, "samples_ms": times,
            "speedup": median["reference"] / median["candidate"],
            "candidate_elements_per_second": batch * length * width * 1000 / median["candidate"],
            "incremental_peak_allocated_bytes": peaks,
            "logical_input_bytes": sum(t.numel() * t.element_size() for t in args),
            "logical_output_bytes": batch * length * width * 4,
        })
    geometric = math.exp(statistics.mean(math.log(row["speedup"]) for row in rows))
    return {
        "kind": "native_cuda_host_inclusive" if cuda else "cpu_interpreter_diagnostic",
        "repeats": repeats, "spec": asdict(spec), "workloads": rows,
        "geometric_speedup": geometric,
        "gpu_latency_throughput_memory": rows if cuda else None,
        "gpu_fitness": None,
        "note": "GPU promotion also requires sanitizers and actual pretrained DFlash inference.",
    }
