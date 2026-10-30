import argparse, csv, os, re, statistics, time
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

def write_summary(rows, path):
    """Average latency/throughput per N, per D and overall, plus fused speedup over unfused.

    Speedup is the geometric mean of per-config ratios unfused_ms / fused_ms
    (only configs where both ran), so one large N does not dominate.
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
        ratios = [lat[(d, n, "unfused")] / ms for (d, n, name), ms in lat.items()
                  if name == "fused" and (d, n, "unfused") in lat]
        row.append(f"{statistics.geometric_mean(ratios):.2f}" if ratios else "")
        return row

    out = [summarize("N", n, [r for r in ok if r[1] == n]) for n in sorted({r[1] for r in ok})]
    out += [summarize("D", d, [r for r in ok if r[0] == d]) for d in sorted({r[0] for r in ok})]
    out.append(summarize("all", "all", ok))

    header = (["scope", "value"]
              + [f"{name}_{m}" for name in impls for m in ("avg_median_ms", "avg_tflops")]
              + ["fused_speedup_vs_unfused"])
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
        pts = [(n, lat[(D, n, "unfused")] / lat[(D, n, "fused")]) for n in Ns
               if (D, n, "fused") in lat and (D, n, "unfused") in lat]
        if pts:
            ax.plot(*zip(*pts), color=PLOT_STYLE["fused"][0], marker="o", markersize=7, linewidth=2)
            for n, s in pts:
                ax.annotate(f"{s:.1f}x", (n, s), textcoords="offset points", xytext=(-6, 6),
                            ha="right", fontsize=9, color="#52514e")
            ax.set_ylim(0, max(s for _, s in pts) * 1.15)
        ax.axhline(1.0, color="#9a9994", linewidth=1, linestyle="--")
        ax.set_ylabel("fused speedup over unfused (x)")

        for ax, title in zip(row, ("Latency", "Throughput", "Fused vs unfused")):
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
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--impls", default="fused,unfused,sdpa")
    p.add_argument("--out", default=None,
                   help="per-config CSV (default: results/bench_forward_sep_<gpu>.csv)")
    p.add_argument("--summary-out", default=None,
                   help="aggregate CSV: averages per N / per D and fused-vs-unfused speedup "
                        "(default: results/bench_forward_agg_<gpu>.csv)")
    p.add_argument("--plot-out", default=None,
                   help="6-panel PNG: latency/throughput/speedup vs N per D "
                        "(default: results/bench_forward_plots_<gpu>.png)")
    args = p.parse_args()

    gpu = re.sub(r"[^A-Za-z0-9]+", "_", torch.cuda.get_device_name()).strip("_")  # e.g. NVIDIA_L4
    args.out = args.out or f"results/bench_forward_sep_{gpu}.csv"
    args.summary_out = args.summary_out or f"results/bench_forward_agg_{gpu}.csv"
    args.plot_out = args.plot_out or f"results/bench_forward_plots_{gpu}.png"

    torch.manual_seed(0)
    rows = []
    for D in (64, 128):
        for N in (128, 512, 2048, 8192):
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
    write_summary(rows, args.summary_out)
    write_plots(rows, args.plot_out)

if __name__ == "__main__":
    main()