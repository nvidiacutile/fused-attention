"""Correctness + timing for the fused backward pass.

    python -m tests.backward.run_correctness_backward                       # default tile 64x64
    python -m tests.backward.run_correctness_backward --tile-m 64 --tile-n 128
    python -m tests.backward.run_correctness_backward --no-bench            # correctness only

Correctness: O, dQ, dK, dV of fused_attention_autograd vs an FP32 PyTorch reference
(same FP16 inputs, same dO). PASS when the relative error ||x - ref|| / ||ref|| <= --rtol for all four.
Timing: forward+backward median ms for fused vs PyTorch SDPA (is_causal=True).
Cases and inputs come from tests/backward/cases_backward.py.
"""
import argparse, csv, math, os, statistics, sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from attention.backward.fused_backward import fused_attention_autograd  # noqa: E402
from tests.backward.cases_backward import (BENCH_B, BENCH_H, BENCH_HEAD_DIMS, BENCH_SEQ_LENS,  # noqa: E402
                                           all_cases, bench_inputs, make_inputs)


def reference(q, k, v):
    """FP32 causal attention with autograd (inputs are FP16 leaves; math in FP32)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    N, D = q.shape[-2], q.shape[-1]
    s = (qf @ kf.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.ones(N, N, device=q.device, dtype=torch.bool).triu(1)
    s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ vf


def rel_err(x, ref):
    x, ref = x.float(), ref.float()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def grads(fn, q, k, v, do):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o = fn(q, k, v)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def time_fwd_bwd(fn, q, k, v, do, warmup=5, reps=20):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    def step():
        q.grad = k.grad = v.grad = None
        fn(q, k, v).backward(do)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); step(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--out", default=None, help="CSV (default: results/correctness/backward/backward_<tile>.csv)")
    a = p.parse_args()
    tm, tn = a.tile_m, a.tile_n
    fused = lambda q, k, v: fused_attention_autograd(q, k, v, tm, tn)
    sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"GPU: {torch.cuda.get_device_name()}  tile {tm}x{tn}")

    rows, fails = [], 0
    print("\n== correctness (relative error vs FP32 reference)")
    print(f"{'B':>2} {'H':>2} {'N':>5} {'D':>4} {'sigma':>5} | {'O':>8} {'dQ':>8} {'dK':>8} {'dV':>8} | result")
    for case in all_cases():
        B, H, N, D, sigma = case.B, case.H, case.N, case.D, case.sigma
        if N % tm or N % tn:
            continue
        (q, k, v), do = make_inputs(case)
        got = grads(fused, q, k, v, do)
        ref = grads(reference, q, k, v, do)
        errs = [rel_err(x, r) for x, r in zip(got, ref)]
        ok = all(e <= a.rtol for e in errs) and all(torch.isfinite(x).all() for x in got)
        fails += not ok
        print(f"{B:>2} {H:>2} {N:>5} {D:>4} {sigma:>5} | " + " ".join(f"{e:8.1e}" for e in errs)
              + f" | {'PASS' if ok else 'FAIL'}")
        rows.append(["correctness", B, H, N, D, sigma, *[f"{e:.3e}" for e in errs], "PASS" if ok else "FAIL",
                     "", "", ""])

    if not a.no_bench:
        print(f"\n== forward+backward time (median ms), B={BENCH_B} H={BENCH_H}")
        print(f"{'D':>4} {'N':>5} | {'fused':>8} {'sdpa':>8} | fused speedup vs sdpa")
        for D in BENCH_HEAD_DIMS:
            for N in BENCH_SEQ_LENS:
                if N % tm or N % tn:
                    continue
                (q, k, v), do = bench_inputs(N, D)
                f_ms = time_fwd_bwd(fused, q, k, v, do)
                s_ms = time_fwd_bwd(sdpa, q, k, v, do)
                print(f"{D:>4} {N:>5} | {f_ms:8.3f} {s_ms:8.3f} | {s_ms / f_ms:.2f}x")
                rows.append(["fwd+bwd", BENCH_B, BENCH_H, N, D, 1.0, "", "", "", "", "", f"{f_ms:.4f}", f"{s_ms:.4f}",
                             f"{s_ms / f_ms:.3f}"])
                torch.cuda.empty_cache()

    out = a.out or os.path.join(REPO_ROOT, "results", "correctness", "backward", f"backward_{tm}x{tn}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "B", "H", "N", "D", "sigma", "relerr_O", "relerr_dQ", "relerr_dK", "relerr_dV",
                    "result", "fused_ms", "sdpa_ms", "fused_speedup_vs_sdpa"])
        w.writerows(rows)
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}   wrote {out}   ({torch.cuda.get_device_name()})")


if __name__ == "__main__":
    main()
