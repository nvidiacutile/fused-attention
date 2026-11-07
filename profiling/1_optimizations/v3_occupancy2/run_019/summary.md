## Run 19 — 2026-11-07 21:03 IST

**Note:** v3 Nsight at D=128, N=8192

### Why (from profiling)

second profiled configuration

### Change

no code change

Code diff vs Run 18 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 18: profiling/kernels/v11_vlate.py
+++ Run 19: profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py
@@ -1,3 +1,3 @@
-# v11: v3 + V tile loaded just before the P@V multiply (after the softmax) instead of at the top of the loop, as in NVIDIA TileGym attention.py
+# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
 import math
 import numpy as np
@@ -35,4 +35,5 @@
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                      order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
@@ -57,6 +58,4 @@
         # 6. update the running sum and the accumulator
         l = l*alpha  + ct.sum(p, axis=-1, keepdims=True)
-        # V is only needed here, so it is loaded after the softmax
-        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
         acc = ct.mma(p.astype(np.float16), v, acc * alpha)
 
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-566184 |
| Git commit | e0a42c7 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 18) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.5 | 0.024 | 1.27× faster | 6.2% slower | 4.6% faster |
| 64 | 2048 | 64x64 | 0.044 | 196.9 | 0.086 | 1.97× faster | 0.9% faster | 43.4% faster |
| 64 | 8192 | 64x64 | 0.447 | 307.6 | 0.557 | 1.25× faster | 0.2% faster | 35.2% faster |
| 128 | 512 | 64x64 | 0.023 | 46.6 | 0.030 | 1.30× faster | 3.6% slower | 3.8% faster |
| 128 | 2048 | 64x64 | 0.075 | 229.5 | 0.117 | 1.57× faster | 0.8% slower | 33.1% faster |
| 128 | 8192 | 64x64 | 0.807 | 340.8 | 1.026 | 1.27× faster | 1.0% faster | 25.5% faster |

### Nsight Compute

Kernel: fused, D=128, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/1_optimizations/v3_occupancy2/run_019/fused_d128_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 810.496000 us |
| Compute (SM) throughput % | 83.795679 % |
| Achieved occupancy % | 24.823703 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

