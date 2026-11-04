"""Forward benchmark for grouped-query attention (GQA).

Q is [B, H, N, D]; K and V are [B, H_kv, N, D] (H_kv divides H). Run once per H_kv, e.g.
    python benchmarks/benchmark_forward_gqa.py --H-kv 2 --tile-m 64 --tile-n 64
fused = attention/gqa/fused_gqa.py, unfused = attention/gqa/unfused_gqa.py (both GQA-aware),
sdpa = torch SDPA with enable_gqa=True. FLOPs are the same as MHA (they depend on H, not H_kv).
Rows are the benchmark_forward.py columns plus H_kv, then fused speedup vs unfused / vs SDPA,
gpu and kernel.
--kernel loads fused_gqa_attention from any kernel file instead of attention/gqa/fused_gqa.py
(e.g. attention/gqa/fused_kernels/v0_baseline.py); --tag names the outputs (default: file name).
Outputs go to results/benchmark/<tag>/<tile>/ by default.
"""
import argparse, csv, importlib.util, os, re, statistics, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from attention.gqa.unfused_gqa import unfused_gqa_attention

fused_gqa_attention = None  # set in main() from --kernel


def load_fused(path):
    """Import fused_gqa_attention from a kernel file without touching attention/gqa/fused_gqa.py."""
    name = "kernel_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_gqa_attention


def sdpa(q, k, v, tm, tn):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

IMPLS = {
    "fused":   lambda q, k, v, tm, tn: fused_gqa_attention(q, k, v, tm, tn),
    "unfused": lambda q, k, v, tm, tn: unfused_gqa_attention(q, k, v, tm, tn),
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

def write_summary(rows, path):
    """Average latency/throughput per N, per D and overall, plus fused speedup over unfused and SDPA.

    Speedup is the geometric mean of per-config ratios unfused_ms / fused_ms (or sdpa_ms / fused_ms),
    only configs where both ran, so one large N does not dominate. >1 means fused is faster.
    """
    ok = [(r[0], r[1], r[4], float(r[7]), float(r[8])) for r in rows if r[7] != ""]
    impls = list(dict.fromkeys(r[2] for r in ok))

    def summarize(scope, value, sub):
        row = [scope, value]
        for name in impls:
            ms = [r[3] for r in sub if r[2] == name]
            tf = [r[4] for r in sub if r[2] == name]
            row += [f"{statistics.mean(ms):.4f}" if ms else "", f"{statistics.mean(tf):.3f}" if tf else ""]
        lat = {(r[0], r[1], r[2]): r[3] for r in sub}
        for other in ("unfused", "sdpa"):
            ratios = [lat[(d, n, other)] / ms for (d, n, name), ms in lat.items()
                      if name == "fused" and (d, n, other) in lat]
            row.append(f"{statistics.geometric_mean(ratios):.2f}" if ratios else "")
        return row

    out = [summarize("N", n, [r for r in ok if r[1] == n]) for n in sorted({r[1] for r in ok})]
    out += [summarize("D", d, [r for r in ok if r[0] == d]) for d in sorted({r[0] for r in ok})]
    out.append(summarize("all", "all", ok))

    header = (["scope", "value"]
              + [f"{name}_{m}" for name in impls for m in ("avg_median_ms", "avg_tflops")]
              + ["fused_speedup_vs_unfused", "fused_speedup_vs_sdpa"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)
    print(f"wrote {path}")

PLOT_STYLE = {  # fixed colour + marker per implementation (marker keeps identity without colour)
    "fused":   ("#2a78d6", "o"),
    "unfused": ("#eb6834", "s"),
    "sdpa":    ("#1baf7a", "^"),
}


def write_plots(rows, path):
    """Six panels, one variable at a time: for each D, latency / throughput / speedup vs N."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    ok = [(r[0], r[1], r[4], float(r[7]), float(r[8])) for r in rows if r[7] != ""]
    Ds = sorted({r[0] for r in ok})
    Ns = sorted({r[1] for r in ok})
    impls = list(dict.fromkeys(r[2] for r in ok))
    lat = {(d, n, i): ms for d, n, i, ms, _ in ok}
    tfl = {(d, n, i): tf for d, n, i, _, tf in ok}

    fig, axes = plt.subplots(len(Ds), 3, figsize=(15, 4.2 * len(Ds)), squeeze=False)
    for row, D in zip(axes, Ds):
        for ax, (metric, ylabel) in zip(row[:2], ((lat, "median latency (ms, log)"), (tfl, "throughput (TFLOP/s)"))):
            for name in impls:
                pts = [(n, metric[(D, n, name)]) for n in Ns if (D, n, name) in metric]
                if pts:
                    color, marker = PLOT_STYLE.get(name, ("#52514e", "D"))
                    ax.plot(*zip(*pts), color=color, marker=marker, markersize=7, linewidth=2, label=name)
            if metric is lat:
                ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False)

        ax = row[2]
        ymax = 1.0
        for other in ("unfused", "sdpa"):
            pts = [(n, lat[(D, n, other)] / lat[(D, n, "fused")]) for n in Ns
                   if (D, n, "fused") in lat and (D, n, other) in lat]
            if pts:
                color, marker = PLOT_STYLE[other]
                ax.plot(*zip(*pts), color=color, marker=marker, markersize=7, linewidth=2,
                        label=f"vs {other}")
                for n, s in pts:
                    ax.annotate(f"{s:.2f}x", (n, s), textcoords="offset points", xytext=(-6, 6),
                                ha="right", fontsize=9, color=color)
                ymax = max(ymax, max(s for _, s in pts))
        ax.set_ylim(0, ymax * 1.15)
        ax.axhline(1.0, color="#9a9994", linewidth=1, linestyle="--")
        ax.set_ylabel("fused speedup (x, >1 = fused faster)")
        ax.legend(frameon=False)

        for ax, title in zip(row, ("Latency", "Throughput", "Fused speedup")):
            ax.set_title(f"{title} vs N  (D={D})", fontsize=11, loc="left")
            ax.set_xscale("log", base=2)
            ax.set_xticks(Ns)
            ax.set_xticklabels([str(n) for n in Ns])
            ax.set_xlabel("sequence length N")
            ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    global fused_gqa_attention
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", default="attention/gqa/fused_gqa.py",
                   help="kernel file with fused_gqa_attention, e.g. attention/gqa/fused_kernels/v0_baseline.py")
    p.add_argument("--tag", default=None, help="name used in output paths (default: kernel file name)")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--H-kv", type=int, default=8, help="K/V heads (must divide --H)")
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--impls", default="fused,unfused,sdpa")
    p.add_argument("--out", default=None,
                   help="per-config CSV (default: results/benchmark/<tag>/<tile>/bench_forward_gqa_sep_<tag>_<tile>_Hkv<H_kv>.csv)")
    p.add_argument("--summary-out", default=None,
                   help="aggregate CSV: averages per N / per D and fused-vs-unfused speedup "
                        "(default: results/benchmark/<tag>/<tile>/bench_forward_gqa_agg_<tag>_<tile>_Hkv<H_kv>.csv)")
    p.add_argument("--plot-out", default=None,
                   help="6-panel PNG: latency/throughput/speedup vs N per D "
                        "(default: results/benchmark/<tag>/<tile>/bench_forward_gqa_plots_<tag>_<tile>_Hkv<H_kv>.png)")
    args = p.parse_args()
    if args.H % args.H_kv != 0:
        p.error(f"--H-kv={args.H_kv} must divide --H={args.H}")

    gpu = re.sub(r"[^A-Za-z0-9]+", "_", torch.cuda.get_device_name()).strip("_")  # e.g. NVIDIA_L4
    fused_gqa_attention = load_fused(args.kernel)
    tag = args.tag or os.path.splitext(os.path.basename(args.kernel))[0]
    tile = f"{args.tile_m}x{args.tile_n}"
    base = f"results/benchmark/{tag}/{tile}"
    print(f"kernel={args.kernel}  tag={tag}  tile={tile}  gpu={gpu}")
    args.out = args.out or f"{base}/bench_forward_gqa_sep_{tag}_{tile}_Hkv{args.H_kv}.csv"
    args.summary_out = args.summary_out or f"{base}/bench_forward_gqa_agg_{tag}_{tile}_Hkv{args.H_kv}.csv"
    args.plot_out = args.plot_out or f"{base}/bench_forward_gqa_plots_{tag}_{tile}_Hkv{args.H_kv}.png"

    torch.manual_seed(0)
    rows = []
    for D in (32, 64, 128):
        for N in (128, 512, 2048, 8192):
            q = torch.randn(args.B, args.H, N, D, device="cuda", dtype=torch.float16)
            k, v = (torch.randn(args.B, args.H_kv, N, D, device="cuda", dtype=torch.float16)
                    for _ in range(2))
            # causal attention FLOPs: 2 matmuls x 2*N*N*D each, halved by the mask
            flops = 4 * args.B * args.H * N * N * D / 2
            for name in args.impls.split(","):
                try:
                    ms, first_ms = time_impl(IMPLS[name], q, k, v, args.tile_m, args.tile_n)
                    tflops = flops / (ms * 1e-3) / 1e12
                    print(f"Hkv={args.H_kv} D={D:<4} N={N:<5} {name:<8} {ms:8.3f} ms  "
                          f"{tflops:6.2f} TFLOP/s  (first call {first_ms:.0f} ms)")
                    rows.append([D, N, args.B, args.H, name, args.tile_m, args.tile_n,
                                 f"{ms:.4f}", f"{tflops:.3f}", f"{first_ms:.1f}", args.H_kv])
                except Exception as e:
                    print(f"Hkv={args.H_kv} D={D:<4} N={N:<5} {name:<8} FAILED: {type(e).__name__}: {e}")
                    rows.append([D, N, args.B, args.H, name, args.tile_m, args.tile_n,
                                 "", "", "", args.H_kv])
                torch.cuda.empty_cache()

    # per-config speedups of fused (blank on rows that are not fused or where a run failed)
    lat = {(r[0], r[1], r[4]): float(r[7]) for r in rows if r[7] != ""}
    for r in rows:
        f_ms = lat.get((r[0], r[1], "fused")) if r[4] == "fused" else None
        for other in ("unfused", "sdpa"):
            o_ms = lat.get((r[0], r[1], other))
            r.append(f"{o_ms / f_ms:.3f}" if f_ms and o_ms else "")
        r += [gpu, args.kernel]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "N", "B", "H", "impl", "tile_m", "tile_n",
                    "median_ms", "tflops", "first_call_ms", "H_kv",
                "fused_speedup_vs_unfused", "fused_speedup_vs_sdpa", "gpu", "kernel"])
        w.writerows(rows)
    print(f"wrote {args.out}")
    write_summary(rows, args.summary_out)
    write_plots(rows, args.plot_out)

if __name__ == "__main__":
    main()