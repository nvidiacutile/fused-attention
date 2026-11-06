## Run 21 — 2026-11-04 17:19 IST

**Note:** v8: v3 + V loaded just before P@V (after the softmax), as in NVIDIA TileGym

### Why (from profiling)

v3 Source page: Sleeping stalls on both K/V loads and an MMA; V is held through the softmax although only P@V needs it

### Change

V load moved from the top of the loop to just before acc = mma(p, v, acc*alpha)

Code diff vs Run 20 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 20: profiling/kernels/v10_exp2fp16.py
+++ Run 21: profiling/kernels/v8_vlate.py
@@ -1,3 +1,3 @@
-# v10: v3 + exp2 computed in FP16 for p (p is cast to FP16 for the P@V multiply anyway); row sum added to l in FP32
+# v8: v3 + V tile loaded just before the P@V multiply (after the softmax) instead of at the top of the loop, as in NVIDIA TileGym attention.py
 import math
 import numpy as np
@@ -35,5 +35,4 @@
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                      order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
@@ -50,13 +49,15 @@
         m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True)*scale)
 
-        # correction factor for the old totals (one value per row, stays FP32)
+        # correction factor for the old totals: e^(m - m_new)
         alpha = ct.exp2(m - m_new, flush_to_zero=True)
 
-        # this tile's numerators: exponent computed in FP32 (one FFMA), exp2 evaluated in FP16
-        p = ct.exp2((s * scale - m_new).astype(np.float16))
+        # this tile's numerators: e^(s - m_new)
+        p = ct.exp2(s * scale - m_new, flush_to_zero=True)   
 
-        # 6. update the running sum (row sums converted to FP32) and the accumulator
-        l = l*alpha + ct.sum(p, axis=-1, keepdims=True).astype(np.float32)
-        acc = ct.mma(p, v, acc * alpha)
+        # 6. update the running sum and the accumulator
+        l = l*alpha  + ct.sum(p, axis=-1, keepdims=True)
+        # V is only needed here, so it is loaded after the softmax
+        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
+        acc = ct.mma(p.astype(np.float16), v, acc * alpha)
 
         m = m_new
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v8_vlate.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-563400 |
| Git commit | 18e8e89 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 20) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 30.3 | 0.022 | 1.24× faster | 0.0% faster | 10.2% faster |
| 64 | 2048 | 64x64 | 0.044 | 195.0 | 0.085 | 1.92× faster | 5.4% faster | 42.9% faster |
| 64 | 8192 | 64x64 | 0.448 | 307.0 | 0.550 | 1.23× faster | 5.7% faster | 35.1% faster |
| 128 | 512 | 64x64 | 0.022 | 48.3 | 0.029 | 1.30× faster | 26.7% faster | 7.1% faster |
| 128 | 2048 | 64x64 | 0.074 | 231.2 | 0.116 | 1.56× faster | 31.1% faster | 33.6% faster |
| 128 | 8192 | 64x64 | 0.815 | 337.3 | 1.019 | 1.25× faster | 28.1% faster | 24.7% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_021/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 446.112000 us |
| Compute (SM) throughput % | 75.446542 % |
| Achieved occupancy % | 24.763981 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

