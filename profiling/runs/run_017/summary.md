## Run 17 — 2026-11-03 20:51 IST

**Note:** v7: v3 + two-phase K/V loop (unmasked tiles without branch/where, then diagonal tiles with mask)

### Why (from profiling)

v3 already masks only diagonal tiles (the blog's K-loop split); test removing the warp-uniform per-iteration branch (branch efficiency 99.96%)

### Change

Single loop with 'if j >= mask_start' split into range(mask_start) unmasked + diagonal tile(s) masked

Code diff vs Run 16 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 16: profiling/kernels/v0_baseline.py
+++ Run 17: profiling/kernels/v7_ksplit.py
@@ -1,3 +1,3 @@
-# v0: baseline = attention/fused.py at commit ea1416a (2026-10-31). Frozen snapshot.
+# v7: v3 + two-phase K/V loop: unmasked tiles (j < mask_start) with no branch/where, then the diagonal tile(s) with the mask
 import math
 import numpy as np
@@ -6,10 +6,11 @@
 
 ConstInt = ct.Constant[int]
+LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
 
 
-@ct.kernel
+@ct.kernel(occupancy=2)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
-    i = ct.bid(0)                 # which Q tile this block owns
-    bh = ct.bid(1)                # which (batch, head)
+    bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
+    i = ct.num_blocks(1) - 1 - ct.bid(1)    # which Q tile; heaviest (last) Q tiles launch first
     b = bh // H                   # goes up by 1 every H [0 to BATCH-1]
     h = bh % H                    # varies from 0 to H-1 in every H
@@ -31,35 +32,36 @@
     mask_start = (i * TILE_M) // TILE_N
 
-    for j in range(num_kv_tiles):
+    # phase 1: K/V tiles fully below the diagonal -> no causal mask, no branch
+    for j in range(mask_start):
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                      order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
         v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
-
-        # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
         s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
-        s = ct.mma(q, kt,s) *scale
-        
-
-        # 2. causal mask only on tiles that straddle/cross the diagonal
-        if j >= mask_start:
-            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
-            s = ct.where(cols > rows, -np.inf, s)
-
-        # 3. new running max: bigger of old m and this tile's row max
-        m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True))
-
-        # correction factor for the old totals: e^(m - m_new)
-        alpha = ct.exp(m - m_new)
-
-        # this tile's numerators: e^(s - m_new)
-        p =  ct.exp(s - m_new)
-
-        # 6. update the running sum and the accumulator
-        l = l*alpha  + ct.sum(p, axis=-1, keepdims=True)
+        s = ct.mma(q, kt, s)
+        m_new = ct.maximum(m, ct.max(s, axis=-1, keepdims=True) * scale)
+        alpha = ct.exp2(m - m_new, flush_to_zero=True)
+        p = ct.exp2(s * scale - m_new, flush_to_zero=True)
+        l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
         acc = ct.mma(p.astype(np.float16), v, acc * alpha)
-
         m = m_new
 
-    out = acc / l
+    # phase 2: tiles that cross the diagonal (j = mask_start .. num_kv_tiles-1) -> apply the causal mask
+    for t in range(num_kv_tiles - mask_start):
+        j = mask_start + t
+        kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
+                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
+        s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
+        s = ct.mma(q, kt, s)
+        cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
+        s = ct.where(cols > rows, -np.inf, s)
+        m_new = ct.maximum(m, ct.max(s, axis=-1, keepdims=True) * scale)
+        alpha = ct.exp2(m - m_new, flush_to_zero=True)
+        p = ct.exp2(s * scale - m_new, flush_to_zero=True)
+        l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
+        acc = ct.mma(p.astype(np.float16), v, acc * alpha)
+        m = m_new
+
+    out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
     ct.store(O, index=(b, h, i, 0), tile=out.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))
 
@@ -71,5 +73,5 @@
     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
     O = torch.empty_like(q)
-    scale = 1.0 / math.sqrt(D)
-    ct.launch(torch.cuda.current_stream(), (N // tile_m, B * H, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
+    scale = LOG2E / math.sqrt(D)      # log2(e) / sqrt(D), folded into one constant
+    ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
     return O
```

### Environment

| Environment | |
|---|---|
| Kernel file | [profiling/kernels/rejected_optimization/v7_ksplit.py](../../kernels/rejected_optimization/v7_ksplit.py) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-562228 |
| Git commit | 11bae55 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 16) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 29.5 | 0.023 | 1.27× faster | - | 7.6% faster |
| 64 | 2048 | 64x64 | 0.044 | 196.4 | 0.086 | 1.96× faster | - | 43.2% faster |
| 64 | 8192 | 64x64 | 0.447 | 307.7 | 0.547 | 1.22× faster | - | 35.2% faster |
| 128 | 512 | 64x64 | 0.023 | 46.7 | 0.029 | 1.27× faster | - | 3.8% faster |
| 128 | 2048 | 64x64 | 0.076 | 225.7 | 0.116 | 1.53× faster | - | 32.0% faster |
| 128 | 8192 | 64x64 | 0.813 | 338.1 | 1.018 | 1.25× faster | - | 24.9% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_017/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 447.168000 us |
| Compute (SM) throughput % | 75.446260 % |
| Achieved occupancy % | 24.774697 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

