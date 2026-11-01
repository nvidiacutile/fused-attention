"""Benchmark the fused cuTile backward pass against naive PyTorch and PyTorch SDPA.

Run from the repo root:
    python -m temp_results.benchmark_backward                    # tile 64x64, all configs
    python -m temp_results.benchmark_backward --tile-m 64 --tile-n 128

Implementations (all causal, FP16 inputs, autograd backward):
  fused  attention.fused_backward.fused_attention_autograd (cuTile forward + recompute backward)
  naive  PyTorch eager: S = QK^T*scale, mask, softmax (FP32), P V. Autograd stores the N x N
         probabilities for the backward; this is the "unfused" training baseline.
  sdpa   torch.nn.functional.scaled_dot_product_attention(is_causal=True)

Per config it measures (median of CUDA-event timings):
  fwd_ms        forward only
  bwd_ms        backward only (forward run untimed before each backward)
  fwd_bwd_ms    forward + backward
  peak_mem_mb   peak extra GPU memory during forward + backward, beyond the inputs and dO
  tflops        fwd+bwd TFLOP/s, counting causal FLOPs: fwd = 2 matmuls, bwd = 2.5 x fwd
                (standard FlashAttention convention; recomputation not counted)
Writes temp_results/backward/bench_backward_<tile>.csv and .md (tables incl. speedups).
"""
import argparse, csv, math, os, statistics

import torch
import torch.nn.functional as F

from attention.fused_backward import fused_attention_autograd


def naive_attention(q, k, v):
    """Unfused causal attention in PyTorch eager; FP16 matmuls, FP32 softmax (as most training code)."""
    N, D = q.shape[-2], q.shape[-1]
    s = (q @ k.transpose(-1, -2)).float() * (1.0 / math.sqrt(D))
    mask = torch.ones(N, N, device=q.device, dtype=torch.bool).triu(1)
    p = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)
    return p.to(q.dtype) @ v


def make_impls(tm, tn):
    return {
        "fused": lambda q, k, v: fused_attention_autograd(q, k, v, tm, tn),
        "naive": naive_attention,
        "sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    }


def _leaves(q, k, v):
    return [t.detach().clone().requires_grad_(True) for t in (q, k, v)]


def _event_ms(fn):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def measure(fn, q, k, v, do, warmup, reps):
    q, k, v = _leaves(q, k, v)

    def clear():
        q.grad = k.grad = v.grad = None

    # warm-up (also compiles the cuTile kernels)
    for _ in range(warmup):
        clear(); fn(q, k, v).backward(do)
    torch.cuda.synchronize()

    fwd, bwd, both = [], [], []
    for _ in range(reps):
        with torch.no_grad():
            fwd.append(_event_ms(lambda: fn(q, k, v)))
    for _ in range(reps):
        clear()
        o = fn(q, k, v)
        torch.cuda.synchronize()
        bwd.append(_event_ms(lambda: o.backward(do)))
        del o
    for _ in range(reps):
        clear()
        both.append(_event_ms(lambda: fn(q, k, v).backward(do)))

    # peak extra memory of one forward + backward
    clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(q, k, v).backward(do)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    return statistics.median(fwd), statistics.median(bwd), statistics.median(both), peak


def fmt(x, spec):
    return "" if x is None else format(x, spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--Ds", default="64,128")
    p.add_argument("--Ns", default="128,512,2048,8192")
    p.add_argument("--impls", default="fused,naive,sdpa")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out-dir", default="temp_results/backward")
    a = p.parse_args()

    tm, tn = a.tile_m, a.tile_n
    impls = make_impls(tm, tn)
    names = a.impls.split(",")
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}  B={a.B} H={a.H}  fused tile {tm}x{tn}")
    print(f"{'D':>4} {'N':>5} {'impl':<6} | {'fwd ms':>8} {'bwd ms':>8} {'fwd+bwd':>8} | {'TFLOP/s':>7} | {'peak MB':>9}")

    rows = []
    for D in map(int, a.Ds.split(",")):
        for N in map(int, a.Ns.split(",")):
            torch.manual_seed(0)
            q, k, v, do = (torch.randn(a.B, a.H, N, D, device="cuda", dtype=torch.float16) for _ in range(4))
            fwd_flops = 4 * a.B * a.H * N * N * D / 2              # 2 matmuls x 2NND, halved by causal mask
            total_flops = fwd_flops * 3.5                          # fwd + bwd (2.5 x fwd)
            for name in names:
                if name == "fused" and (N % tm or N % tn):
                    continue
                try:
                    f_ms, b_ms, fb_ms, mem = measure(impls[name], q, k, v, do, a.warmup, a.reps)
                    tf = total_flops / (fb_ms * 1e-3) / 1e12
                    print(f"{D:>4} {N:>5} {name:<6} | {f_ms:8.3f} {b_ms:8.3f} {fb_ms:8.3f} | {tf:7.2f} | {mem:9.1f}")
                    rows.append(dict(D=D, N=N, impl=name, fwd_ms=f_ms, bwd_ms=b_ms, fwd_bwd_ms=fb_ms,
                                     tflops=tf, peak_mem_mb=mem, status="ok"))
                except torch.cuda.OutOfMemoryError:
                    print(f"{D:>4} {N:>5} {name:<6} | out of memory")
                    rows.append(dict(D=D, N=N, impl=name, status="OOM"))
                except Exception as e:
                    print(f"{D:>4} {N:>5} {name:<6} | FAILED: {type(e).__name__}: {e}")
                    rows.append(dict(D=D, N=N, impl=name, status=f"FAILED: {type(e).__name__}"))
                torch.cuda.empty_cache()
            del q, k, v, do

    os.makedirs(a.out_dir, exist_ok=True)
    tile = f"{tm}x{tn}"
    fields = ["D", "N", "B", "H", "impl", "tile_m", "tile_n", "fwd_ms", "bwd_ms", "fwd_bwd_ms", "tflops",
              "peak_mem_mb", "status", "gpu"]
    csv_path = os.path.join(a.out_dir, f"bench_backward_{tile}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()},
                        "B": a.B, "H": a.H, "tile_m": tm, "tile_n": tn, "gpu": gpu})

    # markdown: times + speedups of fused (other_ms / fused_ms, >1 = fused faster) + memory
    R = {(r["D"], r["N"], r["impl"]): r for r in rows if r["status"] == "ok"}
    cfgs = sorted({(r["D"], r["N"]) for r in rows})

    def get(D, N, impl, key):
        r = R.get((D, N, impl))
        return r[key] if r else None

    def ratio(x, y):
        return x / y if x and y else None

    L = [f"# Backward benchmark (fused tile {tile})", "",
         f"GPU: {gpu}; B={a.B}, H={a.H}; FP16 inputs; causal. Median of {a.reps} runs.  ",
         "Speedup = other_ms / fused_ms (>1 = fused faster). Memory = peak extra GPU memory during fwd+bwd.", "",
         "## Forward + backward", "",
         "| D | N | fused ms | naive ms | SDPA ms | vs naive | vs SDPA | fused TFLOP/s | SDPA TFLOP/s |",
         "|---|---|---|---|---|---|---|---|---|"]
    for D, N in cfgs:
        fu, na, sd = (get(D, N, i, "fwd_bwd_ms") for i in ("fused", "naive", "sdpa"))
        L.append(f"| {D} | {N} | {fmt(fu, '.3f')} | {fmt(na, '.3f') or 'OOM'} | {fmt(sd, '.3f')} | "
                 f"{fmt(ratio(na, fu), '.2f')} | {fmt(ratio(sd, fu), '.2f')} | "
                 f"{fmt(get(D, N, 'fused', 'tflops'), '.1f')} | {fmt(get(D, N, 'sdpa', 'tflops'), '.1f')} |")
    L += ["", "## Backward only", "",
          "| D | N | fused bwd ms | naive bwd ms | SDPA bwd ms | vs naive | vs SDPA | fused bwd / fused fwd |",
          "|---|---|---|---|---|---|---|---|"]
    for D, N in cfgs:
        fu, na, sd = (get(D, N, i, "bwd_ms") for i in ("fused", "naive", "sdpa"))
        L.append(f"| {D} | {N} | {fmt(fu, '.3f')} | {fmt(na, '.3f')} | {fmt(sd, '.3f')} | "
                 f"{fmt(ratio(na, fu), '.2f')} | {fmt(ratio(sd, fu), '.2f')} | "
                 f"{fmt(ratio(fu, get(D, N, 'fused', 'fwd_ms')), '.2f')} |")
    L += ["", "## Peak extra memory (MB)", "",
          "| D | N | fused | naive | SDPA | naive / fused |", "|---|---|---|---|---|---|"]
    for D, N in cfgs:
        fu, na, sd = (get(D, N, i, "peak_mem_mb") for i in ("fused", "naive", "sdpa"))
        L.append(f"| {D} | {N} | {fmt(fu, '.1f')} | {fmt(na, '.1f')} | {fmt(sd, '.1f')} | {fmt(ratio(na, fu), '.1f')} |")
    md_path = os.path.join(a.out_dir, f"bench_backward_{tile}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
