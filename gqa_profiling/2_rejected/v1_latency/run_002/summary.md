## Run 2 — 2026-11-05 02:35 IST

**Note:** v1: latency hint on K/V loads

### Change

Added latency=4 hint on K/V ct.load

Code diff vs Run 1 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 1: attention/gqa/fused_kernels/v0_baseline.py
+++ Run 2: attention/gqa/fused_kernels/v1_latency.py
@@ -1,3 +1,3 @@
-# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
+# v4: v3 + latency hint on K/V loads
 # GQA: K/V have H_kv heads; query head h reads K/V head h // (H // H_kv). Grid and O stay on H query heads.
 import math
@@ -8,4 +8,6 @@
 ConstInt = ct.Constant[int]
 LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
+
+KV_LATENCY = 4
 
 
@@ -36,6 +38,7 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
+                     order=(0, 1, 3, 2), latency=KV_LATENCY).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
+                    latency=KV_LATENCY).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
```

### Environment

| Environment | |
|---|---|
| Kernel file | attention/gqa/fused_kernels/v1_latency.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-563768 |
| Git commit | af2dd5f (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |
| K/V heads H_kv | 2 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 1) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.1 | 0.024 | 1.26× faster | 3.8% slower | 3.8% slower |
| 64 | 2048 | 64x64 | 0.046 | 188.4 | 0.086 | 1.89× faster | 2.5% slower | 2.5% slower |
| 64 | 8192 | 64x64 | 0.446 | 308.2 | 0.548 | 1.23× faster | 0.1% faster | 0.1% faster |
| 128 | 512 | 64x64 | 0.024 | 45.2 | 0.030 | 1.26× faster | 5.8% slower | 5.8% slower |
| 128 | 2048 | 64x64 | 0.075 | 228.9 | 0.117 | 1.56× faster | 1.6% slower | 1.6% slower |
| 128 | 8192 | 64x64 | 0.802 | 342.8 | 1.027 | 1.28× faster | 0.1% faster | 0.1% faster |

### Nsight Compute

Profiling failed: ERR_NVGPUCTRPERM (no permission for GPU counters).

---

