#!/usr/bin/env python3
"""Where one sampling step's time goes on one rank, from a step_profile trace.

    analyze_step_trace.py TRACE.json.gz [TRACE2.json.gz ...]

Reads the Chrome trace actor/step_profile.py writes (torch.profiler, CPU and
CUDA) and prints, per rank:
- the window: first to last GPU activity, GPU busy (union of kernels, copies
  and memsets over every stream) and idle, with idle gaps by size;
- kernel time by category, and how much of NCCL's time overlaps compute;
- the host: kernel launches, their CPU time, and host-blocking calls
  (synchronizations, device-to-host copies, .item()/.tolist() style reads);
- the top kernels by total time.
Stdlib only, so it runs anywhere.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict

CATEGORIES = [  # first match wins, on the lowercased kernel name
    ("nccl", r"nccl"),
    ("attention (sage)", r"qk_int8|sv_f8|attn|flash|fmha|sage"),
    ("int8 gemm (triton/kitchen)", r"int8.*(mm|gemm|linear)|_scaled_mm|matmul_kernel|triton_.*mm|imma|i8"),
    ("gemm (cublas/cutlass bf16/fp16)", r"nvjet|cutlass|gemm|sm\d+_xmma|cublas"),
    ("quantize", r"quant|amax|absmax|per_warp|per_block|per_channel|fp8|e4m3"),
    ("norm/rope/adaln (kitchen)", r"rms|rope|adaln|layer_norm|norm"),
    ("copy/cat/transpose", r"copy|cat|transpose|permute|contiguous|index|gather|scatter|slice"),
    ("elementwise/reduce", r"elementwise|vectorized|reduce|unrolled|silu|gelu|mul|add|tanh|exp"),
]


def category(name: str) -> str:
    low = name.lower()
    for label, pattern in CATEGORIES:
        if re.search(pattern, low):
            return label
    return "other"


def union(intervals):
    total, end_max, merged = 0.0, None, []
    for start, end in sorted(intervals):
        if end_max is None or start > end_max:
            merged.append([start, end])
            end_max = end
        elif end > end_max:
            merged[-1][1] = end
            end_max = end
    for start, end in merged:
        total += end - start
    return total, merged


def overlap(a, b):
    """Total length of the intersection of two merged interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


SYNC = re.compile(r"cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|"
                  r"cudaMemcpy(?!Async)|cudaStreamWaitEvent")
D2H_OPS = re.compile(r"aten::item|aten::_local_scalar_dense|aten::tolist|aten::nonzero|"
                     r"aten::is_nonzero|aten::equal|aten::allclose")


def analyze(path: str) -> None:
    with gzip.open(path) if path.endswith(".gz") else open(path, "rb") as handle:
        events = json.load(handle).get("traceEvents", [])
    gpu, by_cat, by_name, nccl, compute = [], defaultdict(float), defaultdict(lambda: [0, 0.0]), [], []
    launches, launch_us, syncs, d2h = 0, 0.0, defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    memcpy = defaultdict(lambda: [0, 0.0])
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat, name = ev.get("cat", ""), ev.get("name", "")
        ts, dur = float(ev.get("ts", 0)), float(ev.get("dur", 0))
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append((ts, ts + dur))
            label = category(name) if cat == "kernel" else cat
            by_cat[label] += dur
            entry = by_name[name[:120]]
            entry[0] += 1
            entry[1] += dur
            (nccl if label == "nccl" else compute).append((ts, ts + dur))
            if cat == "gpu_memcpy":
                memcpy[name][0] += 1
                memcpy[name][1] += dur
        elif cat == "cuda_runtime" or cat == "cuda_driver":
            if "Launch" in name:
                launches += 1
                launch_us += dur
            if SYNC.search(name):
                syncs[name][0] += 1
                syncs[name][1] += dur
        elif cat == "cpu_op" and D2H_OPS.search(name):
            d2h[name][0] += 1
            d2h[name][1] += dur
    if not gpu:
        print(f"{path}: no GPU activity in the trace")
        return
    start, end = min(s for s, _ in gpu), max(e for _, e in gpu)
    window = end - start
    busy, merged = union(gpu)
    gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
    nccl_busy, nccl_merged = union(nccl)
    compute_busy, compute_merged = union(compute)
    both = overlap(nccl_merged, compute_merged)
    ms = lambda us: f"{us / 1000:9.1f} ms"  # noqa: E731
    print(f"=== {path}")
    print(f"GPU window {ms(window)}  busy {ms(busy)} ({100 * busy / window:.1f}%)  idle {ms(window - busy)}")
    for lo, hi in ((0, 10), (10, 100), (100, 1000), (1000, 10000), (10000, float("inf"))):
        sel = [g for g in gaps if lo <= g < hi]
        hi_s = "inf" if hi == float("inf") else f"{hi:g}"
        print(f"  idle gaps {lo:g}-{hi_s} us: {len(sel):6d}  total {ms(sum(sel))}")
    print(f"NCCL busy {ms(nccl_busy)}; overlapped with compute {ms(both)}; "
          f"exposed {ms(nccl_busy - both)}; compute busy {ms(compute_busy)}")
    print("kernel time by category (sum over streams):")
    for label, dur in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {label:34s} {ms(dur)}  {100 * dur / window:5.1f}% of window")
    print(f"host: {launches} kernel launches, {ms(launch_us)} CPU in launch calls")
    for name, (count, dur) in sorted(syncs.items(), key=lambda kv: -kv[1][1]):
        print(f"  host-blocking {name:40s} x{count:6d} {ms(dur)}")
    for name, (count, dur) in sorted(d2h.items(), key=lambda kv: -kv[1][1]):
        print(f"  device read   {name:40s} x{count:6d} {ms(dur)}")
    for name, (count, dur) in sorted(memcpy.items(), key=lambda kv: -kv[1][1]):
        print(f"  copy          {name:40s} x{count:6d} {ms(dur)}")
    print("top kernels:")
    for name, (count, dur) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"  {ms(dur)} x{count:6d}  [{category(name)}] {name[:100]}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        analyze(arg)
