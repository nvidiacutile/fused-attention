## Run 7 — 2026-11-01 00:40 IST

**Note:** unequalTiles/v0: baseline, no optimizations

### Change

Code diff vs Run 6 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 6: profiling/kernels/tiles/v0_baseline.py
+++ Run 7: profiling/kernels/unequalTiles/v0_baseline.py
@@ -1,3 +1,3 @@
-# tiles/v0: baseline = attention/fused.py at commit 95e7266 (2026-10-31). Frozen snapshot.
+# unequalTiles/v0: baseline = attention/fused.py at commit 9af6cc0 (2026-10-31). Frozen snapshot.
 import math
 import numpy as np
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/unequalTiles/v0_baseline.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-558225 |
| Git commit | b2d5573 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 6) on this GPU | vs baseline (Run 2) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.1 | 0.024 | 1.25× faster | 3.5% faster | 3.8% slower |
| 64 | 2048 | 64x64 | 0.078 | 109.6 | 0.086 | 1.09× faster | 0.0% faster | 2.6% slower |
| 64 | 8192 | 64x64 | 0.691 | 199.0 | 0.548 | 1.26× slower | 0.6% faster | 1.7% faster |
| 128 | 512 | 64x64 | 0.025 | 43.3 | 0.028 | 1.14× faster | 0.4% faster | 1.6% slower |
| 128 | 2048 | 64x64 | 0.113 | 152.0 | 0.116 | 1.03× faster | 0.2% faster | 1.3% slower |
| 128 | 8192 | 64x64 | 1.083 | 253.7 | 1.009 | 1.07× slower | 0.7% faster | 0.5% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_007/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 693.184000 us |
| Compute (SM) throughput % | 45.583798 % |
| Achieved occupancy % | 12.337024 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

