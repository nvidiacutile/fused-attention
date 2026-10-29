import argparse, csv, os, statistics, time
import torch
import torch.nn.functional as F
from attention.fused import fused_attention
from attention.unfused import unfused_attention

def sdpa(q, k, v, tm, tn):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

IMPLS = {
    "fused":   lambda q, k, v, tm, tn: fused_attention(q, k, v, tm, tn),
    "unfused": lambda q, k, v, tm, tn: unfused_attention(q, k, v, tm, tn),
    "sdpa":    sdpa,
}

def time_impl(fn, q, k, v, tm, tn, warmup=10, reps=30):
    # 1) first call = compile + run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(q, k, v, tm, tn)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1000

    # 2) warm-up
    for _ in range(warmup):
        fn(q, k, v, tm, tn)
    torch.cuda.synchronize()

    # 3) timed runs with CUDA events
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(q, k, v, tm, tn)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))   # milliseconds
    return statistics.median(times), first_call_ms

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--impls", default="fused,unfused,sdpa")
    p.add_argument("--out", default="results/bench_forward.csv")
    args = p.parse_args()

    torch.manual_seed(0)
    rows = []
    for D in (64, 128):
        for N in (512, 2048, 8192):
            q, k, v = (torch.randn(args.B, args.H, N, D, device="cuda",
                                   dtype=torch.float16) for _ in range(3))
            # causal attention FLOPs: 2 matmuls x 2*N*N*D each, halved by the mask
            flops = 4 * args.B * args.H * N * N * D / 2
            for name in args.impls.split(","):
                try:
                    ms, first_ms = time_impl(IMPLS[name], q, k, v, args.tile_m, args.tile_n)
                    tflops = flops / (ms * 1e-3) / 1e12
                    print(f"D={D:<4} N={N:<5} {name:<8} {ms:8.3f} ms  "
                          f"{tflops:6.2f} TFLOP/s  (first call {first_ms:.0f} ms)")
                    rows.append([D, N, args.B, args.H, name, args.tile_m, args.tile_n,
                                 f"{ms:.4f}", f"{tflops:.3f}", f"{first_ms:.1f}"])
                except Exception as e:
                    print(f"D={D:<4} N={N:<5} {name:<8} FAILED: {type(e).__name__}: {e}")
                    rows.append([D, N, args.B, args.H, name, args.tile_m, args.tile_n,
                                 "", "", "", ])
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "N", "B", "H", "impl", "tile_m", "tile_n",
                    "median_ms", "tflops", "first_call_ms"])
        w.writerows(rows)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()