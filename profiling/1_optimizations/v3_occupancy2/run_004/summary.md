## Run 4 — 2026-11-01 19:02 IST

**Note:** v3: + occupancy=2

### Why (from profiling)

v2 ncu (D=64,N=8192): latency-bound (compute 58%, memory 49%); 255 registers/thread and 82 KB shared memory/block limit occupancy to 1 block (8 warps) per SM, 16.7% theoretical, 12.3% achieved; 2 theoretical warps per scheduler vs ~5.4 cycles per issued instruction; no eligible warp in 72% of cycles. Est. speedup 42%.

### Change

Added occupancy=2 to @ct.kernel so the compiler targets 2 blocks per SM (<=128 registers/thread).

Code diff vs Run 3 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 3: profiling/kernels/v2_fusedInstructions.py
+++ Run 4: profiling/kernels/v3_occupancy2.py
@@ -1,3 +1,3 @@
-# v2: v1 + exp2 with folded constants (log2(e)/sqrt(D) passed as scale) + flush_to_zero on exp2
+# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel
+@ct.kernel(occupancy=2)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
```

### Environment

| Environment | |
|---|---|
| Kernel file | [profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py](../v3_occupancy2.py) |
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
| Git commit | 1942d7b |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 3) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.6 | 0.024 | 1.27× faster | 0.5% slower | 4.6% faster |
| 64 | 2048 | 64x64 | 0.043 | 197.3 | 0.086 | 1.98× faster | 20.8% faster | 43.5% faster |
| 64 | 8192 | 64x64 | 0.447 | 307.6 | 0.555 | 1.24× faster | 19.8% faster | 35.2% faster |
| 128 | 512 | 64x64 | 0.023 | 46.6 | 0.030 | 1.31× faster | 0.4% faster | 3.8% faster |
| 128 | 2048 | 64x64 | 0.074 | 232.0 | 0.117 | 1.59× faster | 14.4% faster | 33.8% faster |
| 128 | 8192 | 64x64 | 0.808 | 340.1 | 1.028 | 1.27× faster | 14.6% faster | 25.3% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/1_optimizations/v3_occupancy2/run_004/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 446.880000 us |
| Compute (SM) throughput % | 75.539317 % |
| Achieved occupancy % | 24.790441 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

