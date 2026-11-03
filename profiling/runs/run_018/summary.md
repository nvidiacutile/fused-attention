## Run 18 — 2026-11-03 21:49 IST

**Note:** v8: v3 + software-pipelined K/V loop (load K/V and compute S for tile j+1 before softmax/PV of tile j)

### Why (from profiling)

v3: each iteration is one dependency chain (load -> QK^T -> max -> exp2 -> sum -> PV); with 4 warps/scheduler no warp is eligible 73% of cycles

### Change

Loop restructured into prologue / steady state / epilogue so the next tile's loads and QK^T are independent of the current softmax

Code diff vs Run 17 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 17: profiling/kernels/v7_ksplit.py
+++ Run 18: profiling/kernels/v8_pipelined.py
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

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_018/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 451.872000 us |
| Compute (SM) throughput % | 75.533755 % |
| Achieved occupancy % | 24.821054 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

