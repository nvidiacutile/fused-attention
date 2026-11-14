"""Live demo: fused cuTile attention (v3, 64x64 tiles): forward, GQA and backward.
Run from the repo root:  python demo_speed.py

Part 1, forward, all 8 configs (D in {64,128} x N in {128,512,2048,8192}, B=2, H=8):
    time of ours, the unfused cuTile baseline, PyTorch SDPA and NVIDIA TileGym (if installed),
    our speedup over each, and max error vs the FP32 reference.
Part 2, grouped-query attention (H=8 query heads, H_kv=2 K/V heads):
    ours vs unfused GQA and SDPA(enable_gqa=True), plus max error vs the FP32 reference.
Part 3, backward:
    dQ, dK, dV vs an FP32 PyTorch reference (relative error, PASS if <= 2e-2, as in the tests),
    and forward+backward time and peak memory vs naive PyTorch.
Speedups are other_ms / ours_ms (> 1 means ours is faster); averages are geometric means.
"""
import math
import statistics

import torch
import torch.nn.functional as F

from attention.fused.fused import fused_attention
from attention.fused.unfused import unfused_attention
from attention.fused.reference import reference_attention
from attention.gqa.fused_gqa import fused_gqa_attention
from attention.gqa.unfused_gqa import unfused_gqa_attention
from attention.gqa.reference_gqa import reference_gqa_attention
from attention.backward.fused_backward import fused_attention_autograd

try:
    from tilegym.ops.cutile.attention import tile_fmha
except Exception:                                  # TileGym not installed: skip that column
    tile_fmha = None

B, H, H_KV, RTOL = 2, 8, 2, 2e-2
CONFIGS = [(D, N) for D in (64, 128) for N in (128, 512, 2048, 8192)]


def time_ms(fn, warmup=10, reps=30):
    for _ in range(warmup):                        # warm-up (also compiles the kernels)
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def geo(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def randn(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float16)


def naive_attention(q, k, v):
    """Unfused PyTorch: autograd keeps the N x N matrices for the backward."""
    N, D = q.shape[-2], q.shape[-1]
    s = (q @ k.transpose(-1, -2)).float() / math.sqrt(D)
    mask = torch.ones(N, N, device=q.device, dtype=torch.bool).triu(1)
    return torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1).to(q.dtype) @ v


def grads(fn, q, k, v, do):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    fn(q, k, v).backward(do)
    return q.grad, k.grad, v.grad


def rel_err(x, ref):
    return ((x.float() - ref.float()).norm() / ref.float().norm()).item()


def fwd_bwd_ms_and_mb(fn, q, k, v, do):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    def step():
        q.grad = k.grad = v.grad = None
        fn(q, k, v).backward(do)

    ms = time_ms(step, warmup=3, reps=10)
    q.grad = k.grad = v.grad = None
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step(); torch.cuda.synchronize()
    return ms, (torch.cuda.max_memory_allocated() - base) / 2**20


torch.manual_seed(0)
print(f"GPU: {torch.cuda.get_device_name()}   B={B} H={H}  FP16  causal  tile 64x64")
if tile_fmha is None:
    print("(TileGym not installed: TileGym column skipped)")

# ---------------------------------------------------------------- part 1: forward
print("\n== 1. Forward: ours vs unfused, SDPA and TileGym (times in ms)")
print(f"{'D':>4} {'N':>5} | {'ours':>7} {'unfused':>8} {'SDPA':>7} {'TileGym':>8} | "
      f"{'vs unfused':>10} {'vs SDPA':>8} {'vs TileGym':>10} | {'max err':>8}")
sp = {"unfused": [], "sdpa": [], "tilegym": []}
for D, N in CONFIGS:
    q, k, v = randn(B, H, N, D), randn(B, H, N, D), randn(B, H, N, D)
    t = {"ours": time_ms(lambda: fused_attention(q, k, v)),
         "unfused": time_ms(lambda: unfused_attention(q, k, v)),
         "sdpa": time_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))}
    if tile_fmha is not None:
        t["tilegym"] = time_ms(lambda: tile_fmha(q, k, v, scaling=1.0 / math.sqrt(D), is_causal=True))
    for name in sp:
        if name in t:
            sp[name].append(t[name] / t["ours"])
    err = (fused_attention(q, k, v).float() - reference_attention(q, k, v)).abs().max().item()
    tg_ms = f"{t['tilegym']:8.3f}" if "tilegym" in t else f"{'-':>8}"
    tg_x = f"{t['tilegym'] / t['ours']:9.2f}x" if "tilegym" in t else f"{'-':>10}"
    print(f"{D:>4} {N:>5} | {t['ours']:7.3f} {t['unfused']:8.3f} {t['sdpa']:7.3f} {tg_ms} | "
          f"{t['unfused'] / t['ours']:9.2f}x {t['sdpa'] / t['ours']:7.2f}x {tg_x} | {err:8.1e}")
print("average speedup:  " + "   ".join(f"vs {n}: {geo(x):.2f}x" for n, x in sp.items() if x))

# ---------------------------------------------------------------- part 2: GQA
print(f"\n== 2. Grouped-query attention ({H} query heads share {H_KV} K/V heads)")
print(f"{'D':>4} {'N':>5} | {'ours':>7} {'unfused':>8} {'SDPA':>7} | {'vs unfused':>10} {'vs SDPA':>8} | {'max err':>8}")
gsp = {"unfused": [], "sdpa": []}
for D, N in CONFIGS:
    q, k, v = randn(B, H, N, D), randn(B, H_KV, N, D), randn(B, H_KV, N, D)
    ours = time_ms(lambda: fused_gqa_attention(q, k, v))
    unf = time_ms(lambda: unfused_gqa_attention(q, k, v))
    sdpa = time_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
    gsp["unfused"].append(unf / ours); gsp["sdpa"].append(sdpa / ours)
    err = (fused_gqa_attention(q, k, v).float() - reference_gqa_attention(q, k, v)).abs().max().item()
    print(f"{D:>4} {N:>5} | {ours:7.3f} {unf:8.3f} {sdpa:7.3f} | {unf / ours:9.2f}x {sdpa / ours:7.2f}x | {err:8.1e}")
print("average speedup:  " + "   ".join(f"vs {n}: {geo(x):.2f}x" for n, x in gsp.items()))

# ---------------------------------------------------------------- part 3: backward
fused = lambda q, k, v: fused_attention_autograd(q, k, v)
ref32 = lambda q, k, v: reference_attention(q, k, v)            # FP32 maths, autograd

print("\n== 3. Backward: gradients vs FP32 reference, and fwd+bwd vs naive PyTorch")
print(f"{'D':>4} {'N':>5} | {'dQ err':>8} {'dK err':>8} {'dV err':>8} | {'result':>6} | "
      f"{'ours ms':>8} {'naive ms':>9} {'speedup':>7} | {'ours MB':>8} {'naive MB':>9} {'less mem':>8}")
all_pass = True
for D, N in [(64, 2048), (128, 2048), (64, 8192)]:
    q, k, v = randn(B, H, N, D), randn(B, H, N, D), randn(B, H, N, D)
    do = torch.randn_like(q)
    errs = [rel_err(a, b) for a, b in zip(grads(fused, q, k, v, do), grads(ref32, q, k, v, do))]
    ok = max(errs) <= RTOL
    all_pass &= ok
    f_ms, f_mb = fwd_bwd_ms_and_mb(fused, q, k, v, do)
    n_ms, n_mb = fwd_bwd_ms_and_mb(naive_attention, q, k, v, do)
    print(f"{D:>4} {N:>5} | {errs[0]:8.1e} {errs[1]:8.1e} {errs[2]:8.1e} | {'PASS' if ok else 'FAIL':>6} | "
          f"{f_ms:8.2f} {n_ms:9.2f} {n_ms / f_ms:6.1f}x | {f_mb:8.0f} {n_mb:9.0f} {n_mb / f_mb:7.0f}x")
print(f"\nbackward correctness: {'ALL PASS' if all_pass else 'SOME FAILED'} (relative error <= {RTOL})")
