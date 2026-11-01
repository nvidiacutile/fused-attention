## Run 9 — 2026-11-01 19:57 IST

**Note:** v2 (occupancy 1): large-tile check

### Change

Code diff vs Run 8 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 8: profiling/kernels/v3_occupancy2.py
+++ Run 9: profiling/kernels/v2_fusedInstructions.py
@@ -1,3 +1,3 @@
-# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
+# v2: v1 + exp2 with folded constants (log2(e)/sqrt(D) passed as scale) + flush_to_zero on exp2
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel(occupancy=2)
+@ct.kernel
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v2_fusedInstructions.py |
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
| Git commit | 4ca8739 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 8) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.6 | 0.024 | 1.28× faster | 0.0% faster | 4.6% faster |
| 64 | 512 | 64x128 | 0.021 | 25.8 | 0.024 | 1.15× faster | 10.1% slower | - |
| 64 | 512 | 128x64 | 0.023 | 23.3 | 0.024 | 1.04× faster | 6.5% faster | - |
| 64 | 512 | 128x128 | 0.041 | 13.0 | 0.024 | 1.72× slower | 13.1% faster | - |
| 64 | 2048 | 64x64 | 0.056 | 153.9 | 0.086 | 1.53× faster | 28.0% slower | 27.5% faster |
| 64 | 2048 | 64x128 | 0.066 | 129.9 | 0.086 | 1.30× faster | 14.2% slower | - |
| 64 | 2048 | 128x64 | 0.058 | 148.5 | 0.086 | 1.48× faster | 19.8% faster | - |
| 64 | 2048 | 128x128 | 0.138 | 62.3 | 0.086 | 1.61× slower | 20.1% faster | - |
| 64 | 8192 | 64x64 | 0.558 | 246.4 | 0.556 | 1.00× slower | 24.3% slower | 19.1% faster |
| 64 | 8192 | 64x128 | 0.725 | 189.5 | 0.556 | 1.30× slower | 36.1% slower | - |
| 64 | 8192 | 128x64 | 0.508 | 270.6 | 0.556 | 1.10× faster | 6.7% slower | - |
| 64 | 8192 | 128x128 | 1.528 | 90.0 | 0.556 | 2.75× slower | 10.6% faster | - |
| 128 | 512 | 64x64 | 0.023 | 46.5 | 0.030 | 1.28× faster | 0.0% faster | 3.3% faster |
| 128 | 512 | 64x128 | 0.023 | 46.7 | 0.030 | 1.29× faster | 21.2% faster | - |
| 128 | 512 | 128x64 | 0.095 | 11.3 | 0.030 | 3.20× slower | 6.2% faster | - |
| 128 | 512 | 128x128 | 0.105 | 10.2 | 0.030 | 3.55× slower | 8.3% faster | - |
| 128 | 2048 | 64x64 | 0.087 | 198.3 | 0.118 | 1.37× faster | 16.6% slower | 22.6% faster |
| 128 | 2048 | 64x128 | 0.086 | 198.9 | 0.118 | 1.37× faster | 10.7% faster | - |
| 128 | 2048 | 128x64 | 0.370 | 46.4 | 0.118 | 3.13× slower | 12.6% faster | - |
| 128 | 2048 | 128x128 | 0.543 | 31.6 | 0.118 | 4.59× slower | 10.4% faster | - |
| 128 | 8192 | 64x64 | 0.950 | 289.3 | 1.026 | 1.08× faster | 23.1% slower | 12.2% faster |
| 128 | 8192 | 64x128 | 0.917 | 299.6 | 1.026 | 1.12× faster | 2.8% faster | - |
| 128 | 8192 | 128x64 | 4.428 | 62.1 | 1.026 | 4.32× slower | 21.6% faster | - |
| 128 | 8192 | 128x128 | 7.426 | 37.0 | 1.026 | 7.24× slower | 32.4% faster | - |

---

