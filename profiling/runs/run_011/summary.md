## Run 11 — 2026-11-01 20:01 IST

**Note:** v3 at 128x64, D=128: spilling check

### Change

Code diff vs Run 10 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 10: profiling/kernels/v4_occupancy3.py
+++ Run 11: profiling/kernels/v3_occupancy2.py
@@ -1,3 +1,3 @@
-# v4: v2 + occupancy=3 hint (target 3 blocks per SM)
+# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel(occupancy=3)
+@ct.kernel(occupancy=2)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
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
| Git commit | 08a2d41 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 10) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 128 | 8192 | 128x64 | 5.735 | 47.9 | 1.026 | 5.59× slower | - | - |

### Nsight Compute

Kernel: fused, D=128, N=8192, tile 128x64. Command: `ncu --set full`. Report: `profiling/runs/run_011/fused_d128_n8192_128x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 5.222208 ms |
| Compute (SM) throughput % | 15.865002 % |
| Achieved occupancy % | 16.583012 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

