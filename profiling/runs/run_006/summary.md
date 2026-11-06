## Run 6 — 2026-11-01 19:26 IST

**Note:** v5: occupancy=2 + latency=4 on K/V loads

### Why (from profiling)

v3 ncu (D=64,N=8192): waiting for async K/V loads (Sleeping + Long Scoreboard) is ~35% of stall cycles; the compiler cut shared memory to 24.7 KB/block at occupancy=2, leaving room up to ~50 KB/block for deeper prefetching without losing the 2nd block. v4 (occupancy=3) was 5.6% faster but spills (676 KB).

### Change

Added latency=4 to the K and V ct.load calls in the loop (v3 base, occupancy=2), to prefetch further ahead.

Code diff vs Run 5 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 5: profiling/kernels/v4_occupancy3.py
+++ Run 6: profiling/kernels/v5_occupancy2latency4.py
@@ -1,3 +1,3 @@
-# v4: v2 + occupancy=3 hint (target 3 blocks per SM)
+# v5: v3 + latency=4 hint 
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel(occupancy=3)
+@ct.kernel(occupancy=2)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
@@ -34,6 +34,7 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
+                     order=(0, 1, 3, 2), latency=4).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D),
+                    latency=4).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
```

### Environment

| Environment | |
|---|---|
| Kernel file | [profiling/kernels/rejected_optimization/v5_occupancy2latency4.py](../../kernels/rejected_optimization/v5_occupancy2latency4.py) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-558933 |
| Git commit | a45c6e3 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 5) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.5 | 0.024 | 1.26× faster | 6.8% slower | 4.1% faster |
| 64 | 2048 | 64x64 | 0.044 | 196.7 | 0.086 | 1.96× faster | 12.8% faster | 43.2% faster |
| 64 | 8192 | 64x64 | 0.449 | 306.1 | 0.559 | 1.24× faster | 5.4% slower | 34.9% faster |
| 128 | 512 | 64x64 | 0.023 | 46.7 | 0.029 | 1.24× faster | 32.4% faster | 3.8% faster |
| 128 | 2048 | 64x64 | 0.075 | 229.3 | 0.118 | 1.58× faster | 42.9% faster | 33.1% faster |
| 128 | 8192 | 64x64 | 0.813 | 338.0 | 1.026 | 1.26× faster | 28.1% faster | 24.8% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_006/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 446.784000 us |
| Compute (SM) throughput % | 75.447806 % |
| Achieved occupancy % | 24.771900 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

