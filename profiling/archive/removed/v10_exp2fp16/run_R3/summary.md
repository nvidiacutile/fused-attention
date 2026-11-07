## R3 — 2026-11-04 17:18 IST

**Note:** v10: v3 + exp2 evaluated in FP16 for p

### Why (from profiling)

v3 Warp State: ~2.2 cycles per instruction on Math Pipe Throttle; the exp2 unit may be one of the busy pipes

### Change

p = exp2((s*scale - m_new).astype(fp16)); row sums converted to FP32 before adding to l

Code diff vs R2 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- R2: profiling/kernels/v9_lazyrescale.py
+++ R3: profiling/kernels/v10_exp2fp16.py
@@ -1,3 +1,3 @@
-# v9: v3 + lazy rescaling: rescale acc and l only when some row max grows by more than 8 (log2 units); otherwise keep the old max
+# v10: v3 + exp2 computed in FP16 for p (p is cast to FP16 for the P@V multiply anyway); row sum added to l in FP32
 import math
 import numpy as np
@@ -7,5 +7,4 @@
 ConstInt = ct.Constant[int]
 LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
-RESCALE_THRESHOLD = 8.0           # log2 units: p may grow up to 2^8 = 256 before we rescale
 
 
@@ -48,23 +47,18 @@
             s = ct.where(cols > rows, -np.inf, s)
 
-        # 3. candidate running max for this tile
-        m_cand = ct.maximum(m, ct.max(s, axis=-1, keepdims=True) * scale)
+        # 3. new running max: bigger of old m and this tile's row max
+        m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True)*scale)
 
-        # 4. lazy rescaling: only when some row's max grew by more than the threshold.
-        #    Otherwise keep the old (stale) max: p = 2^(s*scale - m) is then at most 2^8,
-        #    and O = acc / l stays exact because acc and l use the same reference max.
-        grow = ct.max(m_cand - m)                     # largest increase over the block's rows
-        if grow > RESCALE_THRESHOLD:
-            alpha = ct.exp2(m - m_cand, flush_to_zero=True)
-            l = l * alpha
-            acc = acc * alpha
-            m = m_cand
+        # correction factor for the old totals (one value per row, stays FP32)
+        alpha = ct.exp2(m - m_new, flush_to_zero=True)
 
-        # 5. this tile's numerators relative to the (possibly stale) max
-        p = ct.exp2(s * scale - m, flush_to_zero=True)
+        # this tile's numerators: exponent computed in FP32 (one FFMA), exp2 evaluated in FP16
+        p = ct.exp2((s * scale - m_new).astype(np.float16))
 
-        # 6. update the running sum and the accumulator (no rescale multiply here)
-        l = l + ct.sum(p, axis=-1, keepdims=True)
-        acc = ct.mma(p.astype(np.float16), v, acc)
+        # 6. update the running sum (row sums converted to FP32) and the accumulator
+        l = l*alpha + ct.sum(p, axis=-1, keepdims=True).astype(np.float32)
+        acc = ct.mma(p, v, acc * alpha)
+
+        m = m_new
 
     out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v10_exp2fp16.py |
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

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (R2) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 30.3 | 0.022 | 1.27× faster | 9.7% faster | 10.2% faster |
| 64 | 2048 | 64x64 | 0.046 | 184.9 | 0.086 | 1.85× faster | 3.9% faster | 39.6% faster |
| 64 | 8192 | 64x64 | 0.474 | 289.6 | 0.546 | 1.15× faster | 0.8% slower | 31.2% faster |
| 128 | 512 | 64x64 | 0.030 | 35.5 | 0.028 | 1.07× slower | 27.3% slower | 26.8% slower |
| 128 | 2048 | 64x64 | 0.108 | 159.2 | 0.116 | 1.07× faster | 36.6% slower | 3.6% faster |
| 128 | 8192 | 64x64 | 1.134 | 242.5 | 1.012 | 1.12× slower | 37.2% slower | 4.8% slower |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/run_R3/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 473.376000 us |
| Compute (SM) throughput % | 71.937240 % |
| Achieved occupancy % | 24.807702 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

