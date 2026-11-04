"""Combine temp_results/<version>/<tile>/bench_forward_sep_*.csv into one markdown comparison.

Run from the repo root after benchmarking each version:
    python temp_results/compare_versions.py
Writes temp_results/comparison.md (and prints it).

Speedups are latency ratios (other_ms / version_ms): >1 means the version is faster.
Averages over configs are geometric means so one large N does not dominate.
"""
import argparse, csv, glob, os, statistics
from collections import defaultdict


def geo(xs):
    return statistics.geometric_mean(xs) if xs else None


def fmt(x, spec=".2f", suffix=""):
    return "" if x is None else f"{x:{spec}}{suffix}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="temp_results")
    p.add_argument("--baseline", default=None, help="version tag to compare against (default: first, e.g. v0_baseline)")
    p.add_argument("--out", default=None, help="default: <root>/comparison.md")
    args = p.parse_args()

    fused = {}                          # (ver, tile, D, N) -> ms
    tflops = {}                         # (ver, tile, D, N) -> TFLOP/s
    other = defaultdict(list)           # (impl, tile, D, N) -> [ms, ...] across version runs
    gpus = set()
    for path in sorted(glob.glob(os.path.join(args.root, "*", "*", "bench_forward_sep_*.csv"))):
        ver = path.split(os.sep)[-3]
        for r in csv.DictReader(open(path)):
            if not r["median_ms"]:
                continue
            tile = f'{r["tile_m"]}x{r["tile_n"]}'
            key = (tile, int(r["D"]), int(r["N"]))
            gpus.add(r.get("gpu", ""))
            if r["impl"] == "fused":
                fused[(ver, *key)] = float(r["median_ms"])
                tflops[(ver, *key)] = float(r["tflops"])
            else:
                other[(r["impl"], *key)].append(float(r["median_ms"]))
    if not fused:
        raise SystemExit(f"no bench_forward_sep_*.csv found under {args.root}/<version>/<tile>/")

    # unfused / SDPA were timed once per version run; use the median across runs
    ref = {k: statistics.median(v) for k, v in other.items()}
    vers = sorted({k[0] for k in fused})
    tiles = sorted({k[1] for k in fused}, key=lambda t: tuple(map(int, t.split("x"))))
    cfgs = sorted({(k[2], k[3]) for k in fused})
    base = args.baseline or vers[0]

    def ratios(ver, tile, against):
        """against: 'unfused' | 'sdpa' | a version tag."""
        out = []
        for D, N in cfgs:
            v = fused.get((ver, tile, D, N))
            o = (ref.get((against, tile, D, N)) if against in ("unfused", "sdpa")
                 else fused.get((against, tile, D, N)))
            if v and o:
                out.append(o / v)
        return out

    L = [f"# Fused attention: version comparison",
         "",
         f"GPU: {', '.join(sorted(g for g in gpus if g)) or 'unknown'}  ",
         f"Configs: D in {sorted({d for d, _ in cfgs})}, N in {sorted({n for _, n in cfgs})}; "
         f"baseline = {base}  ",
         "Speedup = other_ms / version_ms (>1 = version faster). Averages over configs are geometric means.",
         ""]

    # 1. summary per tile x version
    L += ["## Summary per tile and version", "",
          f"| Tile | Version | Avg fused ms | Avg TFLOP/s | vs {base} | vs unfused | vs SDPA |",
          "|---|---|---|---|---|---|---|"]
    for tile in tiles:
        for ver in vers:
            ms = [fused[(ver, tile, D, N)] for D, N in cfgs if (ver, tile, D, N) in fused]
            tf = [tflops[(ver, tile, D, N)] for D, N in cfgs if (ver, tile, D, N) in tflops]
            if not ms:
                continue
            L.append(f"| {tile} | {ver} | {statistics.mean(ms):.4f} | {statistics.mean(tf):.2f} | "
                     f"{fmt(geo(ratios(ver, tile, base)), '.2f', 'x')} | "
                     f"{fmt(geo(ratios(ver, tile, 'unfused')), '.2f', 'x')} | "
                     f"{fmt(geo(ratios(ver, tile, 'sdpa')), '.2f', 'x')} |")
    L.append("")

    # 2. per tile: every config, every version
    for tile in tiles:
        L += [f"## Tile {tile}: fused ms (speedup vs SDPA)", "",
              "| D | N | SDPA ms | unfused ms | " + " | ".join(vers) + " | best |",
              "|---|---|---|---|" + "---|" * len(vers) + "---|"]
        for D, N in cfgs:
            s = ref.get(("sdpa", tile, D, N))
            u = ref.get(("unfused", tile, D, N))
            cells, best = [], None
            for ver in vers:
                v = fused.get((ver, tile, D, N))
                if v is None:
                    cells.append("")
                    continue
                cells.append(f"{v:.4f} ({fmt(s / v if s else None, '.2f', 'x')})")
                if best is None or v < best[1]:
                    best = (ver, v)
            L.append(f"| {D} | {N} | {fmt(s, '.4f')} | {fmt(u, '.4f')} | " + " | ".join(cells)
                     + f" | {best[0] if best else ''} |")
        L.append("")

    # 3. version-over-version change at each config (all tiles averaged)
    L += ["## Step-by-step change (geometric mean over all tiles and configs)", "",
          "| Step | Speedup | Configs faster / total |", "|---|---|---|"]
    for prev, cur in zip(vers, vers[1:]):
        rs = [r for t in tiles for r in ratios(cur, t, prev)]
        if rs:
            L.append(f"| {prev} -> {cur} | {geo(rs):.2f}x | {sum(r > 1 for r in rs)} / {len(rs)} |")
    L.append("")

    # 4. best version + tile per config
    L += ["## Best version and tile per config", "",
          "| D | N | Best | ms | vs SDPA | vs unfused (same tile) |", "|---|---|---|---|---|---|"]
    for D, N in cfgs:
        cands = [(ms, ver, tile) for (ver, tile, d, n), ms in fused.items() if (d, n) == (D, N)]
        ms, ver, tile = min(cands)
        s, u = ref.get(("sdpa", tile, D, N)), ref.get(("unfused", tile, D, N))
        L.append(f"| {D} | {N} | {ver} {tile} | {ms:.4f} | {fmt(s / ms if s else None, '.2f', 'x')} | "
                 f"{fmt(u / ms if u else None, '.2f', 'x')} |")
    L.append("")

    out = args.out or os.path.join(args.root, "comparison.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
