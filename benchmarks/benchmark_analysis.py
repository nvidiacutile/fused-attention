"""Read results/bench_forward.csv and explain it.

    python -m benchmarks.benchmark_analysis [--csv results/bench_forward.csv] [--peak 121]

Prints a comparison table, speedups, memory saved by fusion, and short
observations. Writes results/bench_forward_summary.md and, if matplotlib is
installed, results/bench_forward.png.
"""
import argparse
import csv
import os
from collections import defaultdict

L4_FP16_PEAK_TFLOPS = 121.0   # NVIDIA L4 dense FP16 Tensor Core peak (approx.)


def load(path):
    data = defaultdict(dict)   # (D, N) -> impl -> row
    meta = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["median_ms"]:
                continue   # failed run
            key = (int(r["D"]), int(r["N"]))
            data[key][r["impl"]] = {
                "ms": float(r["median_ms"]),
                "tflops": float(r["tflops"]),
                "first_ms": float(r["first_call_ms"]),
            }
            meta = {"B": int(r["B"]), "H": int(r["H"]),
                    "tile": f'{r["tile_m"]}x{r["tile_n"]}'}
    return data, meta


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/bench_forward.csv")
    p.add_argument("--peak", type=float, default=L4_FP16_PEAK_TFLOPS,
                   help="GPU FP16 peak TFLOP/s for the %%-of-peak column")
    p.add_argument("--out-md", default="results/bench_forward_summary.md")
    p.add_argument("--out-png", default="results/bench_forward.png")
    args = p.parse_args()

    data, meta = load(args.csv)
    if not data:
        print(f"no successful rows in {args.csv}")
        return
    B, H = meta["B"], meta["H"]
    keys = sorted(data, key=lambda k: (k[0], k[1]))
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"# Forward benchmark summary")
    out()
    out(f"B={B}, H={H}, tile={meta['tile']} (fused/unfused), FP16 inputs, causal. "
        f"Peak used for %: {args.peak:.0f} TFLOP/s.")
    out()

    # ---- main table ------------------------------------------------------
    out("## Latency (median ms) and throughput")
    out()
    out("| D | N | unfused ms | fused ms | SDPA ms | fused TFLOP/s | fused % peak "
        "| fused vs unfused | fused vs SDPA |")
    out("|---|---|---|---|---|---|---|---|---|")
    speedups, sdpa_ratios = {}, {}
    for D, N in keys:
        row = data[(D, N)]
        u, f, s = (row.get(x, {}).get("ms") for x in ("unfused", "fused", "sdpa"))
        ft = row.get("fused", {}).get("tflops")
        su = u / f if (u and f) else None
        rs = f / s if (f and s) else None
        speedups[(D, N)], sdpa_ratios[(D, N)] = su, rs
        cell = lambda x, d=3: f"{x:.{d}f}" if x is not None else "-"
        out(f"| {D} | {N} | {cell(u)} | {cell(f)} | {cell(s)} | {cell(ft, 2)} | "
            f"{cell(ft / args.peak * 100 if ft else None, 1)}% | "
            f"{cell(su, 2)}x faster | "
            f"{(cell(rs, 2) + 'x slower') if rs and rs >= 1 else (cell(1 / rs, 2) + 'x faster') if rs else '-'} |")
    out()

    # ---- memory that fusion avoids ---------------------------------------
    out("## Global-memory traffic avoided by fusion")
    out()
    out("Unfused writes S (FP32) and P (FP16), each [B,H,N,N], to global memory "
        "and reads them back. Fused keeps them on-chip.")
    out()
    out("| N | S+P size in HBM (unfused) | Q+K+V+O size |")
    out("|---|---|---|")
    for N in sorted({k[1] for k in keys}):
        sp = B * H * N * N * (4 + 2)
        D = max(d for d, n in keys if n == N)
        qkvo = 4 * B * H * N * D * 2
        out(f"| {N} | {fmt_bytes(sp)} | {fmt_bytes(qkvo)} (D={D}) |")
    out()

    # ---- compile time ----------------------------------------------------
    out("## First-call time (includes JIT compile; not part of latency above)")
    out()
    for impl in ("fused", "unfused", "sdpa"):
        firsts = [data[k][impl]["first_ms"] for k in keys if impl in data[k]]
        if firsts:
            out(f"- {impl}: max {max(firsts):.0f} ms, min {min(firsts):.0f} ms")
    out()

    # ---- automatic observations ------------------------------------------
    out("## Observations")
    out()
    for D in sorted({k[0] for k in keys}):
        ns = sorted(n for d, n in keys if d == D and speedups.get((d, n)))
        if len(ns) >= 2:
            first, last = speedups[(D, ns[0])], speedups[(D, ns[-1])]
            trend = "grows" if last > first * 1.1 else "shrinks" if last < first / 1.1 else "stays roughly flat"
            out(f"- D={D}: fused-vs-unfused speedup {trend} with N "
                f"({first:.2f}x at N={ns[0]} -> {last:.2f}x at N={ns[-1]}). "
                + ("Expected: unfused traffic grows as N^2, so fusion helps more at long sequences."
                   if trend == "grows" else
                   "Unexpected: check that unfused really materializes S and P, and that sizes are large enough."))
    slower = [k for k, r in sdpa_ratios.items() if r and r > 1]
    faster = [k for k, r in sdpa_ratios.items() if r and r <= 1]
    if slower:
        worst = max(slower, key=lambda k: sdpa_ratios[k])
        out(f"- Fused is slower than SDPA in {len(slower)}/{len(sdpa_ratios)} sizes "
            f"(worst {sdpa_ratios[worst]:.2f}x at D={worst[0]}, N={worst[1]}). "
            "This is the gap the optimization phase targets.")
    if faster:
        out(f"- Fused matches or beats SDPA at: "
            + ", ".join(f"D={d} N={n}" for d, n in sorted(faster)) + ".")
    fused_pct = [data[k]["fused"]["tflops"] / args.peak * 100 for k in keys if "fused" in data[k]]
    if fused_pct:
        best = max(fused_pct)
        out(f"- Best fused utilization: {best:.1f}% of peak. "
            + ("Low (<30%): likely latency/occupancy or memory-bound; profile with Nsight Compute."
               if best < 30 else
               "Moderate: compute starting to dominate; tile tuning and exp2/fast-math may help."
               if best < 60 else "High: close to hardware limits."))
    over = [(k, i) for k in keys for i in data[k] if data[k][i]["tflops"] > args.peak]
    if over:
        out(f"- WARNING: {len(over)} result(s) exceed the assumed peak of {args.peak:.0f} TFLOP/s "
            f"(e.g. {over[0][1]} at D={over[0][0][0]}, N={over[0][0][1]}). The peak value is wrong "
            "for this GPU/clock, so treat '% peak' as unreliable; compare against SDPA instead, "
            "or pass --peak with the correct number.")
    small = [k for k in keys if k[1] == min(n for _, n in keys)]
    if small and all("fused" in data[k] and data[k]["fused"]["ms"] < 0.05 for k in small):
        out(f"- At N={small[0][1]} runs are under 0.05 ms: launch overhead dominates, "
            "so differences there say little about the kernel itself.")
    out()

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}")

    # ---- optional plot ---------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install matplotlib)")
        return
    Ds = sorted({k[0] for k in keys})
    fig, axes = plt.subplots(1, len(Ds), figsize=(6 * len(Ds), 4), squeeze=False)
    for ax, D in zip(axes[0], Ds):
        ns = sorted(n for d, n in keys if d == D)
        for impl in ("unfused", "fused", "sdpa"):
            ys = [data[(D, n)].get(impl, {}).get("ms") for n in ns]
            pts = [(n, y) for n, y in zip(ns, ys) if y is not None]
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", label=impl)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length N")
        ax.set_ylabel("median latency (ms, log)")
        ax.set_title(f"D={D}, B={B}, H={H}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
