"""TileGym benchmark: our fused kernel vs NVIDIA TileGym tile_fmha vs PyTorch SDPA (causal, FP16).

Sweep: D in (64, 128) x N in (128, 512, 2048, 8192), B=2, H=8; our kernel runs at 64x64 tiles.
Timing matches benchmarks/benchmark_forward.py: one first call (compile + run), 10 warm-up calls,
then the median of 30 CUDA-event-timed calls.
Speedups are latency ratios (>1 = the first-named impl is faster):
    ours vs TileGym = tilegym_ms / ours_ms
    ours vs SDPA    = sdpa_ms / ours_ms
    TileGym vs SDPA = sdpa_ms / tilegym_ms
Writes a per-config CSV, an aggregate CSV (per N / per D / overall) and a 6-panel plot to --out-dir
(default: results/benchmark/tilegym_<tag>/64x64/, <tag> = kernel file name).

    python benchmarks/benchmark_tilegym.py
    python benchmarks/benchmark_tilegym.py --out-dir results/benchmark/my_tilegym_run
    python benchmarks/benchmark_tilegym.py --kernel attention/fused/fused_kernels/v0_baseline.py
"""
import argparse, csv, importlib.util, math, os, re, statistics, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

try:
    from tilegym.ops.cutile.attention import tile_fmha
except ImportError as e:  # reported in main() so --help still works without TileGym
    tile_fmha, TILEGYM_IMPORT_ERROR = None, e

TILE_M = TILE_N = 64
DS = (64, 128)
NS = (128, 512, 2048, 8192)
IMPLS = ("ours", "tilegym", "sdpa")
PAIRS = (("ours", "tilegym"), ("ours", "sdpa"), ("tilegym", "sdpa"))  # (a, b): speedup of a over b = b_ms / a_ms
LABEL = {"ours": "ours", "tilegym": "TileGym", "sdpa": "SDPA"}
PLOT_STYLE = {"ours": ("#2a78d6", "o"), "tilegym": ("#eb6834", "s"), "sdpa": ("#1baf7a", "^")}
PAIR_STYLE = {("ours", "tilegym"): ("#eb6834", "s", "-"), ("ours", "sdpa"): ("#1baf7a", "^", "-"),
              ("tilegym", "sdpa"): ("#52514e", "D", "--")}
HEADER = ["D", "N", "B", "H", "impl", "tile_m", "tile_n", "median_ms", "tflops", "first_call_ms",
          "max_abs_diff_vs_sdpa", "speedup_vs_tilegym", "speedup_vs_sdpa", "gpu", "kernel"]
FMT = {"median_ms": "{:.4f}", "tflops": "{:.3f}", "first_call_ms": "{:.1f}", "max_abs_diff_vs_sdpa": "{:.2e}",
       "speedup_vs_tilegym": "{:.3f}", "speedup_vs_sdpa": "{:.3f}"}


def load_fused(path):
    """Import fused_attention from a kernel file without touching attention/fused/fused.py."""
    name = "kernel_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_attention


def time_impl(fn, warmup=10, reps=30):
    """Same protocol as benchmark_forward.time_impl: first call, warm-up, median of CUDA-event timings."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1000
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds
    return statistics.median(times), first_call_ms


def geo(xs):
    return statistics.geometric_mean(xs) if xs else None


def write_summary(rows, path):
    """Average latency/throughput per impl per N, per D and overall, plus geometric-mean speedups."""
    ok = [r for r in rows if r["median_ms"] != ""]
    lat = {(r["D"], r["N"], r["impl"]): r["median_ms"] for r in ok}

    def summarize(scope, value, sub):
        row = [scope, value]
        for name in IMPLS:
            ms = [r["median_ms"] for r in sub if r["impl"] == name]
            tf = [r["tflops"] for r in sub if r["impl"] == name]
            row += [f"{statistics.mean(ms):.4f}" if ms else "", f"{statistics.mean(tf):.3f}" if tf else ""]
        cfgs = sorted({(r["D"], r["N"]) for r in sub})
        for a, b in PAIRS:
            s = geo([lat[(d, n, b)] / lat[(d, n, a)] for d, n in cfgs if (d, n, a) in lat and (d, n, b) in lat])
            row.append(f"{s:.3f}" if s else "")
        return row

    out = [summarize("N", n, [r for r in ok if r["N"] == n]) for n in sorted({r["N"] for r in ok})]
    out += [summarize("D", d, [r for r in ok if r["D"] == d]) for d in sorted({r["D"] for r in ok})]
    out.append(summarize("all", "all", ok))
    header = (["scope", "value"] + [f"{name}_{m}" for name in IMPLS for m in ("avg_median_ms", "avg_tflops")]
              + [f"{a}_speedup_vs_{b}" for a, b in PAIRS])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)
    print(f"wrote {path}")


def write_plots(rows, path):
    """One row of panels per D: latency vs N, throughput vs N, speedups vs N."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    ok = [r for r in rows if r["median_ms"] != ""]
    Ds = sorted({r["D"] for r in ok})
    Ns = sorted({r["N"] for r in ok})
    lat = {(r["D"], r["N"], r["impl"]): r["median_ms"] for r in ok}
    tfl = {(r["D"], r["N"], r["impl"]): r["tflops"] for r in ok}
    fig, axes = plt.subplots(len(Ds), 3, figsize=(15, 4.2 * len(Ds)), squeeze=False)
    for row, D in zip(axes, Ds):
        for ax, (metric, ylabel) in zip(row[:2], ((lat, "median latency (ms, log)"), (tfl, "throughput (TFLOP/s)"))):
            for name in IMPLS:
                pts = [(n, metric[(D, n, name)]) for n in Ns if (D, n, name) in metric]
                if pts:
                    color, marker = PLOT_STYLE[name]
                    ax.plot(*zip(*pts), color=color, marker=marker, markersize=7, linewidth=2, label=LABEL[name])
            if metric is lat:
                ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False)

        ax = row[2]
        ymax = 1.0
        for a, b in PAIRS:
            pts = [(n, lat[(D, n, b)] / lat[(D, n, a)]) for n in Ns if (D, n, a) in lat and (D, n, b) in lat]
            if pts:
                color, marker, ls = PAIR_STYLE[(a, b)]
                ax.plot(*zip(*pts), color=color, marker=marker, markersize=7, linewidth=2, linestyle=ls,
                        label=f"{LABEL[a]} vs {LABEL[b]}")
                for n, s in pts:
                    ax.annotate(f"{s:.2f}x", (n, s), textcoords="offset points", xytext=(-6, 6),
                                ha="right", fontsize=9, color=color)
                ymax = max(ymax, max(s for _, s in pts))
        ax.set_ylim(0, ymax * 1.15)
        ax.axhline(1.0, color="#9a9994", linewidth=1, linestyle="--")
        ax.set_ylabel("speedup (x, >1 = first-named faster)")
        ax.legend(frameon=False)

        for ax, title in zip(row, ("Latency", "Throughput", "Speedup")):
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", default="attention/fused/fused_kernels/v3_occupancy2.py",
                   help="our kernel file with fused_attention (default: v3_occupancy2, the best version)")
    p.add_argument("--tag", default=None, help="name used in output file names (default: kernel file name)")
    p.add_argument("--out-dir", default=None, help="output folder (default: results/benchmark/tilegym_<tag>/64x64)")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=8)
    args = p.parse_args()
    if tile_fmha is None:
        p.error(f"cannot import TileGym (tilegym.ops.cutile.attention.tile_fmha): {TILEGYM_IMPORT_ERROR}")

    ours = load_fused(args.kernel)
    tag = args.tag or os.path.splitext(os.path.basename(args.kernel))[0]
    tile = f"{TILE_M}x{TILE_N}"
    out_dir = args.out_dir or f"results/benchmark/tilegym_{tag}/{tile}"
    os.makedirs(out_dir, exist_ok=True)
    sep_path = os.path.join(out_dir, f"bench_tilegym_sep_{tag}_{tile}.csv")
    agg_path = os.path.join(out_dir, f"bench_tilegym_agg_{tag}_{tile}.csv")
    plot_path = os.path.join(out_dir, f"bench_tilegym_plots_{tag}_{tile}.png")
    gpu = torch.cuda.get_device_name()
    print(f"gpu={gpu}  ours={args.kernel} at {tile}  B={args.B} H={args.H}  out_dir={out_dir}")

    rows = []
    for D in DS:
        for N in NS:
            torch.manual_seed(0)
            q, k, v = (torch.randn(args.B, args.H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
            flops = 4 * args.B * args.H * N * N * D / 2  # causal: 2 matmuls x 2*N*N*D, halved by the mask
            fns = {
                "ours": lambda: ours(q, k, v, TILE_M, TILE_N),
                "tilegym": lambda: tile_fmha(q, k, v, scaling=1.0 / math.sqrt(D), is_causal=True),
                "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            }
            ref = fns["sdpa"]().float()  # accuracy reference (note: this also warms SDPA before its first call)
            for name in IMPLS:
                r = dict(D=D, N=N, B=args.B, H=args.H, impl=name,
                         tile_m=TILE_M if name == "ours" else "", tile_n=TILE_N if name == "ours" else "",
                         median_ms="", tflops="", first_call_ms="", max_abs_diff_vs_sdpa="")
                try:
                    ms, first_ms = time_impl(fns[name])
                    err = (fns[name]().float() - ref).abs().max().item()
                    r.update(median_ms=ms, tflops=flops / (ms * 1e-3) / 1e12, first_call_ms=first_ms,
                             max_abs_diff_vs_sdpa=err)
                    print(f"D={D:<4} N={N:<5} {name:<8} {ms:8.4f} ms  {r['tflops']:7.2f} TFLOP/s  "
                          f"(first call {first_ms:.0f} ms)  max|diff| vs SDPA {err:.2e}")
                except Exception as e:
                    print(f"D={D:<4} N={N:<5} {name:<8} FAILED: {type(e).__name__}: {e}")
                rows.append(r)
            del ref
            torch.cuda.empty_cache()

    # per-config speedups: speedup_vs_X on a row = X_ms / this_ms (filled only for the pairs in PAIRS)
    lat = {(r["D"], r["N"], r["impl"]): r["median_ms"] for r in rows if r["median_ms"] != ""}
    for r in rows:
        for other in ("tilegym", "sdpa"):
            a, b = (r["D"], r["N"], r["impl"]), (r["D"], r["N"], other)
            ok = (r["impl"], other) in PAIRS and a in lat and b in lat
            r[f"speedup_vs_{other}"] = lat[b] / lat[a] if ok else ""
        r["gpu"], r["kernel"] = gpu, args.kernel

    with open(sep_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([FMT[c].format(r[c]) if c in FMT and r[c] != "" else r[c] for c in HEADER])
    print(f"wrote {sep_path}")

    print("\nspeedups (>1 = first-named faster):")
    for D in DS:
        for N in NS:
            parts = [f"{LABEL[a]} vs {LABEL[b]} {lat[(D, N, b)] / lat[(D, N, a)]:.2f}x"
                     for a, b in PAIRS if (D, N, a) in lat and (D, N, b) in lat]
            print(f"D={D:<4} N={N:<5} " + "   ".join(parts))

    write_summary(rows, agg_path)
    write_plots(rows, plot_path)


if __name__ == "__main__":
    main()
