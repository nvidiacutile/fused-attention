"""Benchmark NVIDIA TileGym's cuTile FMHA against our fused kernel (v3) and PyTorch SDPA.

Run from the repo root (TileGym installed, see profiling notes):
    python -m temp_results.benchmark_tilegym
    python -m temp_results.benchmark_tilegym --kernel profiling/kernels/v3_occupancy2.py

Same timing as benchmarks/benchmark_forward.py: warm-up, then the median of 30 CUDA-event runs.
TileGym is only called (tilegym.ops.cutile.attention.tile_fmha); none of its code is copied here.
Writes temp_results/tilegym/bench_tilegym.csv and .md.
"""
import argparse, csv, importlib.util, math, os, re, statistics

import torch
import torch.nn.functional as F
from tilegym.ops.cutile.attention import tile_fmha


def load_fused(path):
    name = "kernel_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_attention


def time_ms(fn, warmup=10, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", default="profiling/kernels/v3_occupancy2.py")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--out-dir", default="temp_results/tilegym")
    a = p.parse_args()
    ours = load_fused(a.kernel)
    gpu = torch.cuda.get_device_name()
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {gpu} (compute capability {cc[0]}.{cc[1]})  ours: {a.kernel} at 64x64")

    rows = []
    for D in (64, 128):
        for N in (128, 512, 2048, 8192):
            torch.manual_seed(0)
            q, k, v = (torch.randn(a.B, a.H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
            flops = 4 * a.B * a.H * N * N * D / 2
            impls = {
                "ours": lambda: ours(q, k, v, 64, 64),
                "tilegym": lambda: tile_fmha(q, k, v, scaling=1.0 / math.sqrt(D), is_causal=True),
                "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            }
            ref = impls["sdpa"]().float()
            res = {}
            for name, fn in impls.items():
                err = (fn().float() - ref).abs().max().item()
                ms = time_ms(fn)
                res[name] = ms
                rows.append(dict(D=D, N=N, impl=name, median_ms=ms, tflops=flops / (ms * 1e-3) / 1e12,
                                 max_abs_diff_vs_sdpa=err))
                print(f"D={D:<4} N={N:<5} {name:<8} {ms:8.4f} ms  {flops / (ms * 1e-3) / 1e12:7.1f} TFLOP/s  "
                      f"max|diff| vs SDPA {err:.2e}")
            print(f"            ours vs tilegym: {res['tilegym'] / res['ours']:.2f}x   "
                  f"ours vs sdpa: {res['sdpa'] / res['ours']:.2f}x   tilegym vs sdpa: {res['sdpa'] / res['tilegym']:.2f}x")
            torch.cuda.empty_cache()

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "bench_tilegym.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["D", "N", "impl", "median_ms", "tflops", "max_abs_diff_vs_sdpa", "gpu"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "gpu": gpu})
    t = {(r["D"], r["N"], r["impl"]): r["median_ms"] for r in rows}
    cfgs = sorted({(r["D"], r["N"]) for r in rows})
    L = ["# Our fused kernel vs NVIDIA TileGym FMHA vs PyTorch SDPA", "",
         f"GPU: {gpu}; B={a.B}, H={a.H}; causal FP16; median of 30 runs. Ours: `{a.kernel}` at 64x64.  ",
         "Speedup = other_ms / ours_ms (>1 = ours faster).", "",
         "| D | N | ours ms | TileGym ms | SDPA ms | ours vs TileGym | ours vs SDPA | TileGym vs SDPA |",
         "|---|---|---|---|---|---|---|---|"]
    gm = {"tg": [], "sd": [], "tgsd": []}
    for D, N in cfgs:
        o, tg, sd = t[(D, N, "ours")], t[(D, N, "tilegym")], t[(D, N, "sdpa")]
        gm["tg"].append(tg / o); gm["sd"].append(sd / o); gm["tgsd"].append(sd / tg)
        L.append(f"| {D} | {N} | {o:.4f} | {tg:.4f} | {sd:.4f} | {tg / o:.2f}x | {sd / o:.2f}x | {sd / tg:.2f}x |")
    L.append(f"| **geo. mean** | | | | | **{statistics.geometric_mean(gm['tg']):.2f}x** | "
             f"**{statistics.geometric_mean(gm['sd']):.2f}x** | **{statistics.geometric_mean(gm['tgsd']):.2f}x** |")
    with open(os.path.join(a.out_dir, "bench_tilegym.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
