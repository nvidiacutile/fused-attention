## Run 5 — 2026-11-01 19:17 IST

**Note:** v4: occupancy=3 (instead of 2)

### Why (from profiling)

v3 ncu (D=64,N=8192): occupancy limited only by registers (128/thread -> 2 blocks/SM; shared memory 24.7 KB allows 3); achieved 24.8%, no eligible warp in 73% of cycles, est. speedup 24%. Tensor pipe 75.5% busy; Math Pipe Throttle ~20% of stalls.

### Change

occupancy=3 instead of 2 in @ct.kernel, so the compiler targets 3 blocks per SM (<=85 registers/thread).

Code diff vs Run 4 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 4: profiling/kernels/v3_occupancy2.py
+++ Run 5: profiling/kernels/v4_occupancy3.py
@@ -1,3 +1,3 @@
-# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
+# v4: v2 + occupancy=3 hint (target 3 blocks per SM)
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel(occupancy=2)
+@ct.kernel(occupancy=3)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
```

### Environment

| Environment | |
|---|---|
| Kernel file | [profiling/2_rejected/v4_occupancy3/v4_occupancy3.py](../v4_occupancy3.py) |
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
| Git commit | 57fa9f2 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 4) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 30.4 | 0.023 | 1.28× faster | 5.9% faster | 10.2% faster |
| 64 | 2048 | 64x64 | 0.050 | 171.4 | 0.086 | 1.72× faster | 15.2% slower | 34.9% faster |
| 64 | 8192 | 64x64 | 0.426 | 322.7 | 0.550 | 1.29× faster | 4.7% faster | 38.2% faster |
| 128 | 512 | 64x64 | 0.034 | 31.5 | 0.029 | 1.18× slower | 47.8% slower | 42.3% slower |
| 128 | 2048 | 64x64 | 0.131 | 131.1 | 0.118 | 1.11× slower | 76.9% slower | 17.2% slower |
| 128 | 8192 | 64x64 | 1.131 | 243.0 | 1.023 | 1.11× slower | 40.0% slower | 4.6% slower |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/2_rejected/v4_occupancy3/run_005/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 422.048000 us |
| Compute (SM) throughput % | 82.437313 % |
| Achieved occupancy % | 36.950664 % |
| Registers / thread | 80 register/thread |
| Occupancy limit (registers, blocks/SM) | 3.000000 block |

---

