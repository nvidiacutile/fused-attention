## Run 7 — 2026-11-01 19:35 IST

**Note:** v6: occupancy=2 + latency=8 on K/V loads

### Why (from profiling)

v5 (latency=4) compiled to the same kernel as v3 (24.68 KB shared memory, 128 registers, identical instruction count and time). Testing a higher hint to see whether the compiler prefetches deeper; shared memory has room up to ~50 KB/block at 2 blocks/SM.

### Change

latency=8 instead of 4 on the K and V ct.load calls (otherwise identical to v5/v3).

Code diff vs Run 6 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 6: profiling/kernels/v5_occupancy2latency4.py
+++ Run 7: profiling/kernels/v6_occupancy2latency8.py
@@ -1,3 +1,3 @@
-# v5: v3 + latency=4 hint 
+# v6: v3 + latency=8 hint on the K/V loads (occupancy=2)
 import math
 import numpy as np
@@ -34,7 +34,7 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2), latency=4).reshape((TILE_D, TILE_N))
+                     order=(0, 1, 3, 2), latency=8).reshape((TILE_D, TILE_N))
         v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D),
-                    latency=4).reshape((TILE_N, TILE_D))
+                    latency=8).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
```

### Environment

| Environment | |
|---|---|
| Kernel file | [profiling/2_rejected/v6_occupancy2latency8/v6_occupancy2latency8.py](../v6_occupancy2latency8.py) |
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
| Git commit | ab2ce3f |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 6) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.5 | 0.024 | 1.27× faster | 0.5% faster | 4.6% faster |
| 64 | 2048 | 64x64 | 0.044 | 196.4 | 0.086 | 1.97× faster | 0.0% faster | 43.2% faster |
| 64 | 8192 | 64x64 | 0.441 | 311.9 | 0.553 | 1.26× faster | 1.9% faster | 36.1% faster |
| 128 | 512 | 64x64 | 0.023 | 46.9 | 0.029 | 1.26× faster | 0.4% faster | 4.2% faster |
| 128 | 2048 | 64x64 | 0.075 | 229.5 | 0.118 | 1.58× faster | 0.1% faster | 33.2% faster |
| 128 | 8192 | 64x64 | 0.814 | 337.5 | 1.028 | 1.26× faster | 0.1% slower | 24.7% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/2_rejected/v6_occupancy2latency8/run_007/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 440.320000 us |
| Compute (SM) throughput % | 76.950784 % |
| Achieved occupancy % | 24.777720 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

