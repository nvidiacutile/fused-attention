"""Run NVIDIA TileGym's cuTile FMHA once inside an NVTX range so Nsight Compute profiles only that call."""
import argparse, math
import torch
from tilegym.ops.cutile.attention import tile_fmha

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=2)
p.add_argument("--H", type=int, default=8)
p.add_argument("--N", type=int, default=8192)
p.add_argument("--D", type=int, default=64)
a = p.parse_args()

torch.manual_seed(0)
q, k, v = (torch.randn(a.B, a.H, a.N, a.D, device="cuda", dtype=torch.float16) for _ in range(3))
for _ in range(3):                                   # warm-up + autotune (not profiled)
    tile_fmha(q, k, v, scaling=1.0 / math.sqrt(a.D), is_causal=True)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("tilegym")                # only this call is profiled
tile_fmha(q, k, v, scaling=1.0 / math.sqrt(a.D), is_causal=True)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("done")
