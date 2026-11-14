"""Demo step 2: forward benchmark of the fused kernel (v3, 64x64 tiles).
Run from the repo root:  python demo_forward.py

Columns: time (ms) of ours, the unfused cuTile baseline, PyTorch SDPA and NVIDIA TileGym,
our speedup over each (other_ms / ours_ms, > 1 means ours is faster), and the max absolute
error of our output vs the FP32 PyTorch reference. The last line gives geometric-mean speedups.
"""
import contextlib
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

from attention.fused.fused import fused_attention
from attention.fused.unfused import unfused_attention
from attention.fused.reference import reference_attention

try:
    from tilegym.ops.cutile.attention import tile_fmha
except Exception:                                  # TileGym not installed: skip that column
    tile_fmha = None

KERNEL_FILE = "attention/fused/fused.py (same code as attention/fused/fused_kernels/v3_occupancy2.py)"
B, H, TILE_M, TILE_N = 2, 8, 64, 64
DS, NS = (64, 128), (128, 512, 2048, 8192)
WARMUP, REPS = 10, 30


@contextlib.contextmanager
def quiet():
    """Hide stdout/stderr (TileGym prints its autotuning log on the first call at each size)."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1); os.dup2(null.fileno(), 2)
        try:
            with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])


def time_ms(fn):
    for _ in range(WARMUP):                        # warm-up (also compiles the kernels)
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def geo(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


torch.manual_seed(0)
print(f"GPU:     {torch.cuda.get_device_name()}")
print(f"Kernel:  {KERNEL_FILE}")
print(f"Tile:    {TILE_M} x {TILE_N}  (query tile x key tile)")
print(f"Configs: B = {B}, H = {H}, D in {list(DS)}, N in {list(NS)}  ->  {len(DS) * len(NS)} configs, "
      f"FP16, causal")
print(f"Timing:  median of {REPS} runs after {WARMUP} warm-up runs (CUDA events)")
if tile_fmha is None:
    print("(TileGym not installed: TileGym columns skipped)")
print()

hdr = (f"{'D':>4} {'N':>5} | {'fused ms':>8} {'unfused ms':>10} {'SDPA ms':>8} {'TileGym ms':>10} | "
       f"{'vs unfused':>10} {'vs SDPA':>8} {'vs TileGym':>10} | {'max err':>8}")
print(hdr)
print("-" * len(hdr))

sp = {"unfused": [], "SDPA": [], "TileGym": []}
for D in DS:
    for N in NS:
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
        t = {"fused": time_ms(lambda: fused_attention(q, k, v, TILE_M, TILE_N)),
             "unfused": time_ms(lambda: unfused_attention(q, k, v, TILE_M, TILE_N)),
             "SDPA": time_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))}
        if tile_fmha is not None:
            with quiet():
                t["TileGym"] = time_ms(lambda: tile_fmha(q, k, v, scaling=1.0 / math.sqrt(D), is_causal=True))
        for name in sp:
            if name in t:
                sp[name].append(t[name] / t["fused"])
        err = (fused_attention(q, k, v, TILE_M, TILE_N).float() - reference_attention(q, k, v)).abs().max().item()
        tg_ms = f"{t['TileGym']:10.3f}" if "TileGym" in t else f"{'-':>10}"
        tg_x = f"{t['TileGym'] / t['fused']:9.2f}x" if "TileGym" in t else f"{'-':>10}"
        print(f"{D:>4} {N:>5} | {t['fused']:8.3f} {t['unfused']:10.3f} {t['SDPA']:8.3f} {tg_ms} | "
              f"{t['unfused'] / t['fused']:9.2f}x {t['SDPA'] / t['fused']:7.2f}x {tg_x} | {err:8.1e}")
        del q, k, v

print("-" * len(hdr))
print("Average speedup (geometric mean over all configs):  "
      + "   ".join(f"vs {n}: {geo(x):.2f}x" for n, x in sp.items() if x))
