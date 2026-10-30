"""Record one numbered performance run of the fused kernel, with environment and timestamp.

Each run times fused (and SDPA, optionally unfused) on a fixed set of sizes,
compares against the previous run and the first run on the same GPU, and saves:

    profiling/runs.csv          every measurement of every run (machine-readable)
    profiling/RUNS.md           one section per run: heading, environment, results table
    profiling/runs/run_NNN/     env.json, the run's own table, and Nsight Compute files (--ncu)

Usage (from the repo root, .venv active):

    python -m profiling.track_run --note "fused v1, no optimizations"
    python -m profiling.track_run --note "exp2 + scale Q once" --ncu
    python -m profiling.track_run --tiles 64x64,128x128 --note "tile sweep"

Record an earlier benchmark CSV (e.g. results/bench_forward.csv) as a run without re-running:

    python -m profiling.track_run --import-csv results/bench_forward.csv \
        --timestamp "2026-10-29 16:30" --gpu "NVIDIA L4" --commit 967838a \
        --note "fused v1 (imported from results/bench_forward.csv)"

This script does not modify any other file in the repo.
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_CSV = os.path.join(HERE, "runs.csv")
RUNS_MD = os.path.join(HERE, "RUNS.md")
RUNS_DIR = os.path.join(HERE, "runs")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

CSV_FIELDS = ["run", "timestamp", "gpu", "commit", "note", "D", "N", "B", "H",
              "impl", "tile_m", "tile_n", "median_ms", "tflops"]

# Nsight Compute metrics summarised in RUNS.md when --ncu is used
NCU_METRICS = {
    "gpu__time_duration.sum": "Duration",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM throughput %",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "Compute (SM) throughput %",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved occupancy %",
    "launch__registers_per_thread": "Registers / thread",
    "launch__occupancy_limit_registers": "Occupancy limit (registers, blocks/SM)",
}


# --------------------------------------------------------------------------- helpers
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def git_commit():
    c = sh("git rev-parse --short HEAD")
    if c and sh("git status --porcelain --untracked-files=no"):
        c += " (+ uncommitted changes)"
    return c or "unknown"


def collect_env(args):
    import torch
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": "%d.%d" % torch.cuda.get_device_capability(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").split("\n")[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "commit": git_commit(),
    }
    try:
        from importlib.metadata import version
        env["cuda_tile"] = version("cuda-tile")
    except Exception:
        env["cuda_tile"] = "unknown"
    ncu = sh("ncu --version | tail -1")
    env["nsight_compute"] = ncu or "not found"
    env["B"], env["H"] = args.B, args.H
    return env


def parse_list(s, cast=int):
    return [cast(x) for x in s.split(",") if x]


def parse_tiles(s):
    return [tuple(int(v) for v in t.lower().split("x")) for t in s.split(",") if t]


def read_runs():
    if not os.path.exists(RUNS_CSV):
        return []
    with open(RUNS_CSV) as f:
        return list(csv.DictReader(f))


def next_run_number(rows):
    return max((int(r["run"]) for r in rows), default=0) + 1


# --------------------------------------------------------------------------- measuring
def time_fn(fn, warmup=10, reps=30):
    import torch
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def measure(args, impls, tiles):
    import torch
    import torch.nn.functional as F
    from attention.fused import fused_attention
    unfused_attention = None
    if "unfused" in impls:
        from attention.unfused import unfused_attention

    out = []
    torch.manual_seed(0)
    for D in parse_list(args.Ds):
        for N in parse_list(args.Ns):
            q, k, v = (torch.randn(args.B, args.H, N, D, device="cuda", dtype=torch.float16)
                       for _ in range(3))
            flops = 4 * args.B * args.H * N * N * D / 2
            for impl in impls:
                configs = tiles if impl in ("fused", "unfused") else [("-", "-")]
                for tm, tn in configs:
                    if impl == "fused":
                        fn = lambda: fused_attention(q, k, v, tm, tn)
                    elif impl == "unfused":
                        fn = lambda: unfused_attention(q, k, v, tm, tn)
                    else:
                        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    label = f"{impl} {tm}x{tn}" if tm != "-" else impl
                    try:
                        ms = time_fn(fn)
                        tf = flops / (ms * 1e-3) / 1e12
                        print(f"D={D:<4} N={N:<5} {label:<16} {ms:8.3f} ms  {tf:7.2f} TFLOP/s")
                        out.append(dict(D=D, N=N, B=args.B, H=args.H, impl=impl,
                                        tile_m=tm, tile_n=tn, median_ms=f"{ms:.4f}",
                                        tflops=f"{tf:.3f}"))
                    except Exception as e:
                        print(f"D={D:<4} N={N:<5} {label:<16} FAILED: {type(e).__name__}: {e}")
                    torch.cuda.empty_cache()
            del q, k, v
            torch.cuda.empty_cache()
    return out


def import_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("median_ms"):
                continue
            tm = r.get("tile_m", "-") or "-"
            tn = r.get("tile_n", "-") or "-"
            if r["impl"] == "sdpa":
                tm = tn = "-"
            rows.append(dict(D=int(r["D"]), N=int(r["N"]), B=int(r["B"]), H=int(r["H"]),
                             impl=r["impl"], tile_m=tm, tile_n=tn,
                             median_ms=r["median_ms"], tflops=r["tflops"]))
    return rows


# --------------------------------------------------------------------------- Nsight Compute
def ncu_profile_target(args):
    """Internal mode: run fused once to warm up, then once more for ncu to capture."""
    import torch
    from attention.fused import fused_attention
    D, N = parse_list(args.ncu_size)
    tm, tn = parse_tiles(args.ncu_tile)[0]
    torch.manual_seed(0)
    q, k, v = (torch.randn(args.B, args.H, N, D, device="cuda", dtype=torch.float16)
               for _ in range(3))
    fused_attention(q, k, v, tm, tn)
    fused_attention(q, k, v, tm, tn)
    torch.cuda.synchronize()


def run_ncu(args, run_dir):
    D, N = parse_list(args.ncu_size)
    name = f"fused_d{D}_n{N}_{args.ncu_tile}"
    rep = os.path.join(run_dir, name)
    target = (f"{sys.executable} -m profiling.track_run --_ncu-target "
              f"--ncu-size {args.ncu_size} --ncu-tile {args.ncu_tile} --B {args.B} --H {args.H}")
    cmd = (f"ncu --set full -k regex:{args.ncu_kernel} --launch-skip 1 --launch-count 1 "
           f"-f -o {rep} {target}")
    print(f"\n[ncu] {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    with open(os.path.join(run_dir, f"{name}_ncu_log.txt"), "w") as f:
        f.write(r.stdout + "\n" + r.stderr)
    if not os.path.exists(rep + ".ncu-rep"):
        msg = "ERR_NVGPUCTRPERM (no permission for GPU counters)" \
            if "ERR_NVGPUCTRPERM" in (r.stdout + r.stderr) else "no report produced; see log"
        print(f"[ncu] failed: {msg}")
        return {"status": f"failed: {msg}"}
    sh(f"ncu --import {rep}.ncu-rep --page details > {rep}_details.txt")
    raw = sh(f"ncu --import {rep}.ncu-rep --page raw --csv")
    metrics = {"status": "ok", "report": os.path.relpath(rep + ".ncu-rep", HERE),
               "size": f"D={D}, N={N}, tile {args.ncu_tile}"}
    try:
        lines = list(csv.reader(raw.splitlines()))
        header, units, values = lines[0], lines[1], lines[2]
        for key, label in NCU_METRICS.items():
            if key in header:
                i = header.index(key)
                metrics[label] = f"{values[i]} {units[i]}".strip()
    except Exception as e:
        metrics["parse_error"] = str(e)
    print(f"[ncu] saved {rep}.ncu-rep")
    return metrics


# --------------------------------------------------------------------------- reporting
def key_of(r):
    return (int(r["D"]), int(r["N"]), r["impl"], str(r["tile_m"]), str(r["tile_n"]))


def find_ref(history, gpu, run, which):
    """Run number of the previous / first run on the same GPU, or None."""
    nums = sorted({int(r["run"]) for r in history if r["gpu"] == gpu and int(r["run"]) < run})
    if not nums:
        return None
    return nums[-1] if which == "prev" else nums[0]


def change(new_ms, old_ms):
    if old_ms is None:
        return "-"
    pct = (old_ms - new_ms) / old_ms * 100
    word = "faster" if pct >= 0 else "slower"
    return f"{abs(pct):.1f}% {word}"


def build_markdown(run, ts, env, note, rows, history, ncu):
    gpu = env["gpu"]
    prev = find_ref(history, gpu, run, "prev")
    base = find_ref(history, gpu, run, "first")
    hist = {(int(r["run"]),) + key_of(r): float(r["median_ms"]) for r in history}

    L = [f"## Run {run} — {ts} IST", ""]
    if note:
        L += [f"**Note:** {note}", ""]
    L += ["| Environment | |", "|---|---|"]
    order = ["gpu", "compute_capability", "gpu_memory_gb", "driver", "torch", "torch_cuda",
             "cuda_tile", "python", "nsight_compute", "host", "commit", "B", "H"]
    names = {"gpu": "GPU", "compute_capability": "Compute capability",
             "gpu_memory_gb": "GPU memory (GB)", "driver": "NVIDIA driver", "torch": "PyTorch",
             "torch_cuda": "PyTorch CUDA", "cuda_tile": "cuda-tile", "python": "Python",
             "nsight_compute": "Nsight Compute", "host": "Host", "commit": "Git commit",
             "B": "Batch B", "H": "Heads H"}
    for k in order:
        if k in env:
            L.append(f"| {names[k]} | {env[k]} |")
    L.append("")

    sdpa = {(r["D"], r["N"]): float(r["median_ms"]) for r in rows if r["impl"] == "sdpa"}
    unf = {(r["D"], r["N"], r["tile_m"], r["tile_n"]): float(r["median_ms"])
           for r in rows if r["impl"] == "unfused"}
    head = "| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA |"
    sep = "|---|---|---|---|---|---|---|"
    if unf:
        head += " Fused vs unfused |"
        sep += "---|"
    head += f" vs previous run{f' ({prev})' if prev else ''} | vs first run{f' ({base})' if base else ''} on this GPU |"
    sep += "---|---|"
    L += ["### Results (fused kernel, median of 30 runs)", "", head, sep]
    for r in sorted((r for r in rows if r["impl"] == "fused"),
                    key=lambda r: (r["D"], r["N"], int(r["tile_m"]))):
        ms = float(r["median_ms"])
        s = sdpa.get((r["D"], r["N"]))
        vs_sdpa = "-" if s is None else (f"{ms / s:.2f}× slower" if ms > s else f"{s / ms:.2f}× faster")
        line = (f"| {r['D']} | {r['N']} | {r['tile_m']}x{r['tile_n']} | {ms:.3f} | "
                f"{float(r['tflops']):.1f} | {'-' if s is None else f'{s:.3f}'} | {vs_sdpa} |")
        if unf:
            u = unf.get((r["D"], r["N"], r["tile_m"], r["tile_n"]))
            line += f" {'-' if u is None else f'{u / ms:.2f}× faster'} |"
        k = key_of(r)
        p = hist.get((prev,) + k) if prev else None
        b = hist.get((base,) + k) if base else None
        line += f" {change(ms, p)} | {change(ms, b)} |"
        L.append(line)
    L.append("")
    if prev is None:
        L += ["_First run on this GPU: it is the baseline for later runs._", ""]

    if ncu:
        L += ["### Nsight Compute", ""]
        if ncu.get("status") != "ok":
            L += [f"Profiling {ncu.get('status')}.", ""]
        else:
            L += [f"Kernel: fused, {ncu['size']}. Report: `profiling/{ncu['report']}`", "",
                  "| Metric | Value |", "|---|---|"]
            for label in NCU_METRICS.values():
                if label in ncu:
                    L.append(f"| {label} | {ncu[label]} |")
            L.append("")
    L.append("---")
    L.append("")
    return "\n".join(L)


def append_results(run, ts, env, note, rows, history, ncu):
    run_dir = os.path.join(RUNS_DIR, f"run_{run:03d}")
    os.makedirs(run_dir, exist_ok=True)
    md = build_markdown(run, ts, env, note, rows, history, ncu)

    new_file = not os.path.exists(RUNS_CSV)
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(dict(run=run, timestamp=ts, gpu=env["gpu"], commit=env.get("commit", ""),
                            note=note, **r))

    if not os.path.exists(RUNS_MD):
        with open(RUNS_MD, "w") as f:
            f.write("# Performance runs\n\nOne section per run, newest at the bottom. "
                    "Times are IST. Changes compare the same size and tile on the same GPU.\n\n")
    with open(RUNS_MD, "a") as f:
        f.write(md + "\n")
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(run_dir, "env.json"), "w") as f:
        json.dump({"run": run, "timestamp": ts, "note": note, "env": env, "ncu": ncu}, f, indent=2)
    print("\n" + md)
    print(f"saved run {run}: {os.path.relpath(RUNS_MD)}, {os.path.relpath(RUNS_CSV)}, "
          f"{os.path.relpath(run_dir)}/")


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--note", default="", help="what changed in this run (shown in the heading)")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--Ds", default="64,128")
    p.add_argument("--Ns", default="512,2048,8192")
    p.add_argument("--tiles", default="64x64", help="fused tile configs, e.g. 64x64,128x128")
    p.add_argument("--with-unfused", action="store_true", help="also time the unfused baseline")
    p.add_argument("--ncu", action="store_true", help="also profile one size with Nsight Compute")
    p.add_argument("--ncu-size", default="64,8192", help="D,N for --ncu (default 64,8192)")
    p.add_argument("--ncu-tile", default="64x64")
    p.add_argument("--ncu-kernel", default="fused", help="regex for the kernel name")
    p.add_argument("--import-csv", help="record an existing benchmark CSV as a run instead of measuring")
    p.add_argument("--timestamp", help="for --import-csv: 'YYYY-MM-DD HH:MM' (IST)")
    p.add_argument("--gpu", help="for --import-csv: GPU name")
    p.add_argument("--commit", help="for --import-csv: git commit of the measured code")
    p.add_argument("--_ncu-target", dest="ncu_target", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.ncu_target:
        ncu_profile_target(args)
        return

    history = read_runs()
    run = next_run_number(history)

    if args.import_csv:
        rows = import_rows(args.import_csv)
        ts = args.timestamp or dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
        env = {"gpu": args.gpu or "unknown", "commit": args.commit or "unknown",
               "B": rows[0]["B"] if rows else "", "H": rows[0]["H"] if rows else "",
               "source": args.import_csv}
        append_results(run, ts, env, args.note, rows, history, None)
        return

    ts = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
    env = collect_env(args)
    print(f"Run {run} — {ts} IST — {env['gpu']} — commit {env['commit']}\n")
    impls = ["fused", "sdpa"] + (["unfused"] if args.with_unfused else [])
    rows = measure(args, impls, parse_tiles(args.tiles))
    ncu = None
    if args.ncu:
        run_dir = os.path.join(RUNS_DIR, f"run_{run:03d}")
        os.makedirs(run_dir, exist_ok=True)
        ncu = run_ncu(args, run_dir)
    append_results(run, ts, env, args.note, rows, history, ncu)


if __name__ == "__main__":
    main()
