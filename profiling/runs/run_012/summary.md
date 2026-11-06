## Run 12 — 2026-11-01 20:21 IST

**Note:** v2 (occupancy 1) at 128x128, D=128 N=8192: large-tile profile

### Why (from profiling)

Large tiles were 4-14x slower than 64x64 at D=128 in the tile x occupancy sweep (Runs 8-10); suspected register spilling.

### Change

No code change; tile 128x128.

Code diff vs Run 11 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 11: profiling/kernels/v3_occupancy2.py
+++ Run 12: profiling/kernels/v2_fusedInstructions.py
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
| Kernel file | [profiling/kernels/v2_fusedInstructions.py](../../kernels/v2_fusedInstructions.py) |
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
| Git commit | 6b1b160 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 11) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 128 | 8192 | 128x128 | 7.412 | 37.1 | 1.025 | 7.23× slower | - | - |

### Nsight Compute

Kernel: fused, D=128, N=8192, tile 128x128. Command: `ncu --set full`. Report: `profiling/runs/run_012/fused_d128_n8192_128x128.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 7.295360 ms |
| Compute (SM) throughput % | 12.809709 % |
| Achieved occupancy % | 12.443358 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

