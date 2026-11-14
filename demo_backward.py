"""Demo: forward + backward benchmark of the fused kernel (v3 forward + recompute backward, 64x64 tiles).
Run from the repo root:  python demo_backward.py

Each implementation runs its own forward and its own backward (one training step for attention):
  fused  attention/backward/fused_backward.py: fused_attention_autograd
  naive  PyTorch eager QK^T, mask, softmax (FP32), PV; autograd keeps the N x N matrices
  SDPA   torch scaled_dot_product_attention(is_causal=True) and its own backward
Columns: forward+backward time (ms), our speedup over naive and SDPA, peak extra GPU memory (MB)
during one forward+backward, how many times less memory than naive, and the largest relative
error of dQ, dK, dV vs an FP32 PyTorch reference.
"""
import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F

from attention.backward.fused_backward import fused_attention_autograd
from attention.fused.reference import reference_attention

p = argparse.ArgumentParser()
p.add_argument("--summary", default=None, help="write the average speedups to this JSON file")
args = p.parse_args()

KERNEL_FILE = "attention/backward/fused_backward.py (v3 forward + dK/dV and dQ backward kernels)"
B, H, TILE_M, TILE_N = 2, 8, 64, 64
DS, NS = (64, 128), (128, 512, 2048, 8192)
WARMUP, REPS = 3, 10

impls = {
    "fused": lambda q, k, v: fused_attention_autograd(q, k, v, TILE_M, TILE_N),
    "naive": None,                                       # set below
    "SDPA": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
}


def naive_attention(q, k, v):
    N, D = q.shape[-2], q.shape[-1]
    s = (q @ k.transpose(-1, -2)).float() / math.sqrt(D)
    mask = torch.ones(N, N, device=q.device, dtype=torch.bool).triu(1)
    return torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1).to(q.dtype) @ v


impls["naive"] = naive_attention


def leaves(q, k, v):
    return [t.detach().clone().requires_grad_(True) for t in (q, k, v)]


def fwd_bwd_ms_and_mb(fn, q, k, v, do):
    q, k, v = leaves(q, k, v)

    def step():
        q.grad = k.grad = v.grad = None
        fn(q, k, v).backward(do)

    for _ in range(WARMUP):                              # warm-up (also compiles the kernels)
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); step(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    q.grad = k.grad = v.grad = None
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step(); torch.cuda.synchronize()
    return statistics.median(ts), (torch.cuda.max_memory_allocated() - base) / 2**20


def grads(fn, q, k, v, do):
    q, k, v = leaves(q, k, v)
    fn(q, k, v).backward(do)
    return q.grad, k.grad, v.grad


def rel_err(x, ref):
    return ((x.float() - ref.float()).norm() / ref.float().norm()).item()


def geo(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


torch.manual_seed(0)
print(f"GPU:     {torch.cuda.get_device_name()}")
print(f"Kernel:  {KERNEL_FILE}")
print(f"Tile:    {TILE_M} x {TILE_N}  (query tile x key tile)")
print(f"Configs: B = {B}, H = {H}, D in {list(DS)}, N in {list(NS)}  ->  {len(DS) * len(NS)} configs, "
      f"FP16, causal")
print(f"Timing:  forward + backward, median of {REPS} runs after {WARMUP} warm-up runs (CUDA events)")
print("Grad err: max over dQ, dK, dV of ||ours - ref|| / ||ref||, ref = FP32 PyTorch\n")

hdr = (f"{'D':>4} {'N':>5} | {'fused ms':>8} {'naive ms':>8} {'SDPA ms':>8} | {'vs naive':>8} {'vs SDPA':>8} | "
       f"{'fused MB':>8} {'naive MB':>8} {'less mem':>8} | {'grad err':>8}")
print(hdr)
print("-" * len(hdr))

sp = {"naive": [], "SDPA": []}
mem_ratios, max_grad_err = [], 0.0
for D in DS:
    for N in NS:
        q, k, v, do = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(4))
        r = {name: fwd_bwd_ms_and_mb(fn, q, k, v, do) for name, fn in impls.items()}
        gerr = max(rel_err(a, b) for a, b in zip(grads(impls["fused"], q, k, v, do),
                                                 grads(reference_attention, q, k, v, do)))
        max_grad_err = max(max_grad_err, gerr)
        f_ms, f_mb = r["fused"]
        for name in sp:
            sp[name].append(r[name][0] / f_ms)
        mem_ratios.append(r["naive"][1] / f_mb)
        print(f"{D:>4} {N:>5} | {f_ms:8.3f} {r['naive'][0]:8.3f} {r['SDPA'][0]:8.3f} | "
              f"{r['naive'][0] / f_ms:7.2f}x {r['SDPA'][0] / f_ms:7.2f}x | "
              f"{f_mb:8.0f} {r['naive'][1]:8.0f} {r['naive'][1] / f_mb:7.1f}x | {gerr:8.1e}")
        del q, k, v, do
        torch.cuda.empty_cache()

avg = {n: geo(x) for n, x in sp.items()}
print("-" * len(hdr))
print("Average speedup (geometric mean over all configs):  "
      + "   ".join(f"vs {n}: {s:.2f}x" for n, s in avg.items())
      + f"   |   memory: up to {max(mem_ratios):.0f}x less than naive")

if args.summary:
    with open(args.summary, "w") as f:
        json.dump({"vs_baseline": avg["naive"], "vs_sdpa": avg["SDPA"], "vs_tilegym": None,
                   "max_vs_baseline": max(sp["naive"]), "max_mem_ratio": max(mem_ratios),
                   "max_err": max_grad_err, "min_vs_sdpa": min(sp["SDPA"]), "max_vs_sdpa": max(sp["SDPA"])}, f)
