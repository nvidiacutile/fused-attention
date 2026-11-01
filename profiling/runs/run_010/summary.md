## Run 10 — 2026-11-01 19:57 IST

**Note:** v4 (occupancy 3): small-tile check

### Change

Code diff vs Run 9 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 9: profiling/kernels/v2_fusedInstructions.py
+++ Run 10: profiling/kernels/v4_occupancy3.py
@@ -1,3 +1,3 @@
-# v2: v1 + exp2 with folded constants (log2(e)/sqrt(D) passed as scale) + flush_to_zero on exp2
+# v4: v2 + occupancy=3 hint (target 3 blocks per SM)
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel
+@ct.kernel(occupancy=3)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v4_occupancy3.py |
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
| Git commit | 4ca8739 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 9) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 32x64 | 0.020 | 27.5 | 0.024 | 1.22× faster | - | - |
| 64 | 512 | 64x32 | 0.021 | 26.1 | 0.024 | 1.17× faster | - | - |
| 64 | 512 | 64x64 | 0.019 | 28.4 | 0.024 | 1.26× faster | 0.5% slower | 4.1% faster |
| 64 | 2048 | 32x64 | 0.061 | 141.1 | 0.086 | 1.42× faster | - | - |
| 64 | 2048 | 64x32 | 0.056 | 154.1 | 0.086 | 1.55× faster | - | - |
| 64 | 2048 | 64x64 | 0.050 | 172.8 | 0.086 | 1.73× faster | 10.9% faster | 35.5% faster |
| 64 | 8192 | 32x64 | 0.681 | 201.8 | 0.560 | 1.22× slower | - | - |
| 64 | 8192 | 64x32 | 0.458 | 300.2 | 0.560 | 1.22× faster | - | - |
| 64 | 8192 | 64x64 | 0.428 | 320.9 | 0.560 | 1.31× faster | 23.2% faster | 37.9% faster |
| 128 | 512 | 32x64 | 0.040 | 27.2 | 0.030 | 1.31× slower | - | - |
| 128 | 512 | 64x32 | 0.027 | 39.3 | 0.030 | 1.10× faster | - | - |
| 128 | 512 | 64x64 | 0.034 | 31.7 | 0.030 | 1.13× slower | 46.8% slower | 41.8% slower |
| 128 | 2048 | 32x64 | 0.187 | 91.8 | 0.118 | 1.59× slower | - | - |
| 128 | 2048 | 64x32 | 0.097 | 176.9 | 0.118 | 1.21× faster | - | - |
| 128 | 2048 | 64x64 | 0.132 | 130.4 | 0.118 | 1.12× slower | 52.1% slower | 17.7% slower |
| 128 | 8192 | 32x64 | 2.526 | 108.8 | 0.977 | 2.59× slower | - | - |
| 128 | 8192 | 64x32 | 0.844 | 325.7 | 0.977 | 1.16× faster | - | - |
| 128 | 8192 | 64x64 | 1.086 | 253.0 | 0.977 | 1.11× slower | 14.3% slower | 0.4% slower |

---

