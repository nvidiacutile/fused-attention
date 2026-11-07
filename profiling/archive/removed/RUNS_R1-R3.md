## R1 — 2026-11-03 21:49 IST

**Note:** v8: v3 + software-pipelined K/V loop (load K/V and compute S for tile j+1 before softmax/PV of tile j)

### Why (from profiling)

v3: each iteration is one dependency chain (load -> QK^T -> max -> exp2 -> sum -> PV); with 4 warps/scheduler no warp is eligible 73% of cycles

### Change

Loop restructured into prologue / steady state / epilogue so the next tile's loads and QK^T are independent of the current softmax

Code diff vs Run 17 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 17: profiling/kernels/v7_ksplit.py
+++ R1: profiling/kernels/v8_pipelined.py
@@ -1,3 +1,3 @@
-# v7: v3 + two-phase K/V loop: unmasked tiles (j < mask_start) with no branch/where, then the diagonal tile(s) with the mask
+# v8: v3 + software-pipelined K/V loop: load K/V and compute S for tile j+1 before the softmax/PV of tile j (independent work in each iteration)
 import math
 import numpy as np
@@ -32,34 +32,41 @@
     mask_start = (i * TILE_M) // TILE_N
 
-    # phase 1: K/V tiles fully below the diagonal -> no causal mask, no branch
-    for j in range(mask_start):
-        kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
-        s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
-        s = ct.mma(q, kt, s)
-        m_new = ct.maximum(m, ct.max(s, axis=-1, keepdims=True) * scale)
+    # ---- prologue: scores for the first K/V tile
+    kt = ct.load(K, index=(b, h, 0, 0), shape=(1, 1, TILE_D, TILE_N),
+                 order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
+    v_cur = ct.load(V, index=(b, h, 0, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
+    s_cur = ct.mma(q, kt, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))
+
+    # ---- steady state: tile j's softmax + P@V overlaps with tile j+1's loads and Q@K^T
+    for j in range(num_kv_tiles - 1):
+        # (a) next tile: independent of the current softmax, so it can run while (b) waits
+        kt_n = ct.load(K, index=(b, h, 0, j + 1), shape=(1, 1, TILE_D, TILE_N),
+                       order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
+        v_n = ct.load(V, index=(b, h, j + 1, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
+        s_n = ct.mma(q, kt_n, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))
+
+        # (b) current tile j: mask (diagonal only), online softmax, P@V
+        if j >= mask_start:
+            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
+            s_cur = ct.where(cols > rows, -np.inf, s_cur)
+        m_new = ct.maximum(m, ct.max(s_cur, axis=-1, keepdims=True) * scale)
         alpha = ct.exp2(m - m_new, flush_to_zero=True)
-        p = ct.exp2(s * scale - m_new, flush_to_zero=True)
+        p = ct.exp2(s_cur * scale - m_new, flush_to_zero=True)
         l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
-        acc = ct.mma(p.astype(np.float16), v, acc * alpha)
+        acc = ct.mma(p.astype(np.float16), v_cur, acc * alpha)
         m = m_new
 
-    # phase 2: tiles that cross the diagonal (j = mask_start .. num_kv_tiles-1) -> apply the causal mask
-    for t in range(num_kv_tiles - mask_start):
-        j = mask_start + t
-        kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
-        s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
-        s = ct.mma(q, kt, s)
-        cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
-        s = ct.where(cols > rows, -np.inf, s)
-        m_new = ct.maximum(m, ct.max(s, axis=-1, keepdims=True) * scale)
-        alpha = ct.exp2(m - m_new, flush_to_zero=True)
-        p = ct.exp2(s * scale - m_new, flush_to_zero=True)
-        l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
-        acc = ct.mma(p.astype(np.float16), v, acc * alpha)
-        m = m_new
+        s_cur = s_n
+        v_cur = v_n
+
+    # ---- epilogue: last K/V tile (always on the diagonal, so always masked)
+    j_last = num_kv_tiles - 1
+    cols = j_last * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
+    s_cur = ct.where(cols > rows, -np.inf, s_cur)
+    m_new = ct.maximum(m, ct.max(s_cur, axis=-1, keepdims=True) * scale)
+    alpha = ct.exp2(m - m_new, flush_to_zero=True)
+    p = ct.exp2(s_cur * scale - m_new, flush_to_zero=True)
+    l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
+    acc = ct.mma(p.astype(np.float16), v_cur, acc * alpha)
 
     out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v8_pipelined.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-562313 |
| Git commit | 266174a |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 17) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 30.6 | 0.022 | 1.24× faster | 3.3% faster | 10.7% faster |
| 64 | 2048 | 64x64 | 0.044 | 193.5 | 0.084 | 1.90× faster | 1.6% slower | 42.3% faster |
| 64 | 8192 | 64x64 | 0.452 | 304.2 | 0.549 | 1.22× faster | 1.1% slower | 34.5% faster |
| 128 | 512 | 64x64 | 0.033 | 32.6 | 0.028 | 1.17× slower | 43.5% slower | 38.1% slower |
| 128 | 2048 | 64x64 | 0.114 | 150.5 | 0.116 | 1.02× faster | 49.9% slower | 2.0% slower |
| 128 | 8192 | 64x64 | 1.164 | 236.1 | 1.012 | 1.15× slower | 43.2% slower | 7.6% slower |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/run_R1/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 451.872000 us |
| Compute (SM) throughput % | 75.533755 % |
| Achieved occupancy % | 24.821054 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

## R2 — 2026-11-04 17:18 IST

**Note:** v9: v3 + lazy rescaling (rescale acc and l only when a row max grows by > 8 in log2 units)

### Why (from profiling)

v3 Warp State: ~3 of 11 cycles per instruction wait on the warp's own previous result (softmax chain); acc*alpha rescales the whole accumulator every tile

### Change

Rescale moved under 'if max(m_cand - m) > 8'; p and l use the stale max otherwise; acc += P.V without the alpha multiply

Code diff vs R1 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- R1: profiling/kernels/v8_pipelined.py
+++ R2: profiling/kernels/v9_lazyrescale.py
@@ -1,3 +1,3 @@
-# v8: v3 + software-pipelined K/V loop: load K/V and compute S for tile j+1 before the softmax/PV of tile j (independent work in each iteration)
+# v9: v3 + lazy rescaling: rescale acc and l only when some row max grows by more than 8 (log2 units); otherwise keep the old max
 import math
 import numpy as np
@@ -7,4 +7,5 @@
 ConstInt = ct.Constant[int]
 LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
+RESCALE_THRESHOLD = 8.0           # log2 units: p may grow up to 2^8 = 256 before we rescale
 
 
@@ -32,41 +33,38 @@
     mask_start = (i * TILE_M) // TILE_N
 
-    # ---- prologue: scores for the first K/V tile
-    kt = ct.load(K, index=(b, h, 0, 0), shape=(1, 1, TILE_D, TILE_N),
-                 order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-    v_cur = ct.load(V, index=(b, h, 0, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
-    s_cur = ct.mma(q, kt, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))
+    for j in range(num_kv_tiles):
+        kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
+                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
 
-    # ---- steady state: tile j's softmax + P@V overlaps with tile j+1's loads and Q@K^T
-    for j in range(num_kv_tiles - 1):
-        # (a) next tile: independent of the current softmax, so it can run while (b) waits
-        kt_n = ct.load(K, index=(b, h, 0, j + 1), shape=(1, 1, TILE_D, TILE_N),
-                       order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-        v_n = ct.load(V, index=(b, h, j + 1, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
-        s_n = ct.mma(q, kt_n, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))
+        # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
+        s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
+        s = ct.mma(q, kt,s) 
+        
 
-        # (b) current tile j: mask (diagonal only), online softmax, P@V
+        # 2. causal mask only on tiles that straddle/cross the diagonal
         if j >= mask_start:
             cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
-            s_cur = ct.where(cols > rows, -np.inf, s_cur)
-        m_new = ct.maximum(m, ct.max(s_cur, axis=-1, keepdims=True) * scale)
-        alpha = ct.exp2(m - m_new, flush_to_zero=True)
-        p = ct.exp2(s_cur * scale - m_new, flush_to_zero=True)
-        l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
-        acc = ct.mma(p.astype(np.float16), v_cur, acc * alpha)
-        m = m_new
+            s = ct.where(cols > rows, -np.inf, s)
 
-        s_cur = s_n
-        v_cur = v_n
+        # 3. candidate running max for this tile
+        m_cand = ct.maximum(m, ct.max(s, axis=-1, keepdims=True) * scale)
 
-    # ---- epilogue: last K/V tile (always on the diagonal, so always masked)
-    j_last = num_kv_tiles - 1
-    cols = j_last * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
-    s_cur = ct.where(cols > rows, -np.inf, s_cur)
-    m_new = ct.maximum(m, ct.max(s_cur, axis=-1, keepdims=True) * scale)
-    alpha = ct.exp2(m - m_new, flush_to_zero=True)
-    p = ct.exp2(s_cur * scale - m_new, flush_to_zero=True)
-    l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
-    acc = ct.mma(p.astype(np.float16), v_cur, acc * alpha)
+        # 4. lazy rescaling: only when some row's max grew by more than the threshold.
+        #    Otherwise keep the old (stale) max: p = 2^(s*scale - m) is then at most 2^8,
+        #    and O = acc / l stays exact because acc and l use the same reference max.
+        grow = ct.max(m_cand - m)                     # largest increase over the block's rows
+        if grow > RESCALE_THRESHOLD:
+            alpha = ct.exp2(m - m_cand, flush_to_zero=True)
+            l = l * alpha
+            acc = acc * alpha
+            m = m_cand
+
+        # 5. this tile's numerators relative to the (possibly stale) max
+        p = ct.exp2(s * scale - m, flush_to_zero=True)
+
+        # 6. update the running sum and the accumulator (no rescale multiply here)
+        l = l + ct.sum(p, axis=-1, keepdims=True)
+        acc = ct.mma(p.astype(np.float16), v, acc)
 
     out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v9_lazyrescale.py |
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
| Git commit | 18e8e89 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (R1) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.020 | 27.5 | 0.022 | 1.15× faster | 11.4% slower | 0.5% faster |
| 64 | 2048 | 64x64 | 0.048 | 177.7 | 0.085 | 1.76× faster | 9.0% slower | 37.1% faster |
| 64 | 8192 | 64x64 | 0.471 | 291.9 | 0.550 | 1.17× faster | 4.2% slower | 31.7% faster |
| 128 | 512 | 64x64 | 0.024 | 45.1 | 0.029 | 1.21× faster | 27.9% faster | 0.4% faster |
| 128 | 2048 | 64x64 | 0.079 | 217.5 | 0.116 | 1.47× faster | 30.8% faster | 29.4% faster |
| 128 | 8192 | 64x64 | 0.826 | 332.8 | 1.016 | 1.23× faster | 29.0% faster | 23.7% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/run_R2/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 473.280000 us |
| Compute (SM) throughput % | 69.500007 % |
| Achieved occupancy % | 24.821079 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

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

