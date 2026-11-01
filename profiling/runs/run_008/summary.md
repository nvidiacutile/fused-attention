## Run 8 — 2026-11-01 19:39 IST

**Note:** v3 (occupancy=2): tile sweep

### Change

Code diff vs Run 7 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 7: profiling/kernels/v6_occupancy2latency8.py
+++ Run 8: profiling/kernels/v3_occupancy2.py
@@ -1,3 +1,3 @@
-# v6: v3 + latency=8 hint on the K/V loads (occupancy=2)
+# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
 import math
 import numpy as np
@@ -34,7 +34,6 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2), latency=8).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D),
-                    latency=8).reshape((TILE_N, TILE_D))
+                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v3_occupancy2.py |
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
| Git commit | ab2ce3f (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Tile 32×64

| D | N | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs 64×64 |
|---|---|---|---|---|---|---|
| 64 | 512 | 0.019 | 27.8 | 0.024 | 1.24× faster | +3% |
| 64 | 2048 | 0.072 | 119.0 | 0.086 | 1.20× faster | +66% |
| 64 | 8192 | 0.793 | 173.4 | 0.555 | 1.43× slower | +77% |
| 128 | 512 | 0.032 | 33.4 | 0.030 | 1.08× slower | +39% |
| 128 | 2048 | 0.153 | 112.5 | 0.117 | 1.30× slower | +106% |
| 128 | 8192 | 1.928 | 142.6 | 1.030 | 1.87× slower | +150% |

### Tile 64×32

| D | N | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs 64×64 |
|---|---|---|---|---|---|---|
| 64 | 512 | 0.019 | 28.4 | 0.024 | 1.26× faster | +1% |
| 64 | 2048 | 0.050 | 173.0 | 0.086 | 1.74× faster | +14% |
| 64 | 8192 | 0.498 | 276.0 | 0.555 | 1.12× faster | +11% |
| 128 | 512 | 0.023 | 46.5 | 0.030 | 1.29× faster | +0% |
| 128 | 2048 | 0.081 | 212.5 | 0.117 | 1.45× faster | +9% |
| 128 | 8192 | 0.865 | 317.6 | 1.030 | 1.19× faster | +12% |

### Tile 64×64

| D | N | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs 64×64 |
|---|---|---|---|---|---|---|
| 64 | 512 | 0.019 | 28.6 | 0.024 | 1.27× faster | — |
| 64 | 2048 | 0.044 | 196.9 | 0.086 | 1.98× faster | — |
| 64 | 8192 | 0.449 | 306.2 | 0.555 | 1.24× faster | — |
| 128 | 512 | 0.023 | 46.5 | 0.030 | 1.29× faster | — |
| 128 | 2048 | 0.074 | 231.3 | 0.117 | 1.58× faster | — |
| 128 | 8192 | 0.772 | 356.0 | 1.030 | 1.33× faster | — |

### Tile 64×128

| D | N | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs 64×64 |
|---|---|---|---|---|---|---|
| 64 | 512 | 0.019 | 28.4 | 0.024 | 1.26× faster | +1% |
| 64 | 2048 | 0.058 | 148.5 | 0.086 | 1.49× faster | +33% |
| 64 | 8192 | 0.533 | 257.9 | 0.555 | 1.04× faster | +19% |
| 128 | 512 | 0.029 | 36.7 | 0.030 | 1.02× faster | +26% |
| 128 | 2048 | 0.097 | 177.5 | 0.117 | 1.21× faster | +30% |
| 128 | 8192 | 0.944 | 291.3 | 1.030 | 1.09× faster | +22% |

### Tile 128×64

| D | N | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs 64×64 |
|---|---|---|---|---|---|---|
| 64 | 512 | 0.025 | 21.7 | 0.024 | 1.03× slower | +31% |
| 64 | 2048 | 0.072 | 119.0 | 0.086 | 1.20× faster | +66% |
| 64 | 8192 | 0.476 | 288.6 | 0.555 | 1.17× faster | +6% |
| 128 | 512 | 0.101 | 10.7 | 0.030 | 3.39× slower | +336% |
| 128 | 2048 | 0.423 | 40.6 | 0.117 | 3.61× slower | +470% |
| 128 | 8192 | 5.646 | 48.7 | 1.030 | 5.48× slower | +631% |

### Tile 128×128

| D | N | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs 64×64 |
|---|---|---|---|---|---|---|
| 64 | 512 | 0.048 | 11.3 | 0.024 | 1.99× slower | +153% |
| 64 | 2048 | 0.172 | 49.8 | 0.086 | 2.00× slower | +295% |
| 64 | 8192 | 1.709 | 80.4 | 0.555 | 3.08× slower | +281% |
| 128 | 512 | 0.115 | 9.4 | 0.030 | 3.86× slower | +396% |
| 128 | 2048 | 0.606 | 28.4 | 0.117 | 5.17× slower | +715% |
| 128 | 8192 | 10.993 | 25.0 | 1.030 | 10.67× slower | +1324% |

---

