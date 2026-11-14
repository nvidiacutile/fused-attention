"""Live demo: fused cuTile attention (v3, 64x64 tiles) forward and backward.
Run from the repo root:  python demo_speed.py

Part 1, forward, all 8 configs (D in {64,128} x N in {128,512,2048,8192}, B=2, H=8):
    our time vs PyTorch SDPA, speedup, and max error vs the FP32 reference.
Part 2, backward:
    dQ, dK, dV vs an FP32 PyTorch reference (relative error, PASS if <= 2e-2, as in the tests),
    and forward+backward time and peak memory vs naive PyTorch (autograd through QK^T, softmax, PV).
"""
import math
import statistics

import torch
import torch.nn.functional as F

from attention.fused.fused import fused_attention
from attention.fused.reference import reference_attention
from attention.backward.fused_backward import fused_attention_autograd

B, H, RTOL = 2, 8, 2e-2


def time_ms(fn, warmup=10, reps=30):
    for _ in range(warmup):                       # warm-up (also compiles the kernels)
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def rand(N, D, grad=False):
    return [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=grad) for _ in range(3)]


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
print(f"GPU: {torch.cuda.get_device_name()}   B={B} H={H}  FP16  causal  tile 64x64\n")

# ---------------------------------------------------------------- part 1: forward
print("== Forward: ours vs PyTorch SDPA")
print(f"{'D':>4} {'N':>5} | {'ours ms':>8} {'SDPA ms':>8} | {'speedup':>7} | {'max err vs FP32':>15}")
speedups = []
for D in (64, 128):
    for N in (128, 512, 2048, 8192):
        q, k, v = rand(N, D)
        ours = time_ms(lambda: fused_attention(q, k, v))
        sdpa = time_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        err = (fused_attention(q, k, v).float() - reference_attention(q, k, v)).abs().max().item()
        speedups.append(sdpa / ours)
        print(f"{D:>4} {N:>5} | {ours:8.3f} {sdpa:8.3f} | {sdpa / ours:6.2f}x | {err:15.2e}")
geo = math.exp(sum(map(math.log, speedups)) / len(speedups))
print(f"geometric-mean speedup over SDPA: {geo:.2f}x\n")

# ---------------------------------------------------------------- part 2: backward
fused = lambda q, k, v: fused_attention_autograd(q, k, v)
ref32 = lambda q, k, v: reference_attention(q, k, v)          # FP32 maths, autograd

print("== Backward: gradients vs FP32 reference, and fwd+bwd vs naive PyTorch")
print(f"{'D':>4} {'N':>5} | {'dQ err':>8} {'dK err':>8} {'dV err':>8} | {'result':>6} | "
      f"{'ours ms':>8} {'naive ms':>9} {'speedup':>7} | {'ours MB':>8} {'naive MB':>9} {'less mem':>8}")
all_pass = True
for D, N in [(64, 2048), (128, 2048), (64, 8192)]:
    q, k, v = rand(N, D)
    do = torch.randn_like(q)
    errs = [rel_err(a, b) for a, b in zip(grads(fused, q, k, v, do), grads(ref32, q, k, v, do))]
    ok = max(errs) <= RTOL
    all_pass &= ok
    f_ms, f_mb = fwd_bwd_ms_and_mb(fused, q, k, v, do)
    n_ms, n_mb = fwd_bwd_ms_and_mb(naive_attention, q, k, v, do)
    print(f"{D:>4} {N:>5} | {errs[0]:8.1e} {errs[1]:8.1e} {errs[2]:8.1e} | {'PASS' if ok else 'FAIL':>6} | "
          f"{f_ms:8.2f} {n_ms:9.2f} {n_ms / f_ms:6.1f}x | {f_mb:8.0f} {n_mb:9.0f} {n_mb / f_mb:7.0f}x")
print(f"\nbackward correctness: {'ALL PASS' if all_pass else 'SOME FAILED'} (relative error <= {RTOL})")
