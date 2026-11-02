## Run 15 — 2026-11-02 20:16 IST

**Note:** v0 128x128, D=128 N=8192: baseline spill check

### Change

Code diff vs Run 14 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 14: profiling/kernels/v2_fusedInstructions.py
+++ Run 15: profiling/kernels/v0_baseline.py
@@ -1,3 +1,3 @@
-# v2: v1 + exp2 with folded constants (log2(e)/sqrt(D) passed as scale) + flush_to_zero on exp2
+# v0: baseline = attention/fused.py at commit ea1416a (2026-10-31). Frozen snapshot.
 import math
 import numpy as np
@@ -6,11 +6,10 @@
 
 ConstInt = ct.Constant[int]
-LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
 
 
 @ct.kernel
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
-    bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
-    i = ct.num_blocks(1) - 1 - ct.bid(1)    # which Q tile; heaviest (last) Q tiles launch first
+    i = ct.bid(0)                 # which Q tile this block owns
+    bh = ct.bid(1)                # which (batch, head)
     b = bh // H                   # goes up by 1 every H [0 to BATCH-1]
     h = bh % H                    # varies from 0 to H-1 in every H
@@ -39,5 +38,5 @@
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
         s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
-        s = ct.mma(q, kt,s) 
+        s = ct.mma(q, kt,s) *scale
         
 
@@ -48,11 +47,11 @@
 
         # 3. new running max: bigger of old m and this tile's row max
-        m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True)*scale)
+        m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True))
 
         # correction factor for the old totals: e^(m - m_new)
-        alpha = ct.exp2(m - m_new, flush_to_zero=True)
+        alpha = ct.exp(m - m_new)
 
         # this tile's numerators: e^(s - m_new)
-        p = ct.exp2(s * scale - m_new, flush_to_zero=True)   
+        p =  ct.exp(s - m_new)
 
         # 6. update the running sum and the accumulator
@@ -62,5 +61,5 @@
         m = m_new
 
-    out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
+    out = acc / l
     ct.store(O, index=(b, h, i, 0), tile=out.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))
 
@@ -72,5 +71,5 @@
     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
     O = torch.empty_like(q)
-    scale = LOG2E / math.sqrt(D)      # log2(e) / sqrt(D), folded into one constant
-    ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
+    scale = 1.0 / math.sqrt(D)
+    ct.launch(torch.cuda.current_stream(), (N // tile_m, B * H, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
     return O
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v0_baseline.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-560722 |
| Git commit | d7d74ea |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 14) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.020 | 27.2 | 0.023 | 1.16× faster | - | 0.5% slower |
| 64 | 2048 | 64x64 | 0.079 | 108.5 | 0.086 | 1.09× faster | - | 2.9% slower |
| 64 | 8192 | 64x64 | 0.700 | 196.3 | 0.552 | 1.27× slower | - | 1.6% slower |
| 128 | 512 | 64x64 | 0.024 | 45.0 | 0.029 | 1.20× faster | - | 0.0% faster |
| 128 | 2048 | 64x64 | 0.114 | 150.6 | 0.117 | 1.03× faster | - | 2.0% slower |
| 128 | 8192 | 64x64 | 1.095 | 251.0 | 1.023 | 1.07× slower | - | 1.2% slower |

### Nsight Compute

Kernel: fused, D=128, N=8192, tile 128x128. Command: `ncu --set full`. Report: `profiling/runs/run_015/fused_d128_n8192_128x128.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 6.154624 ms |
| Compute (SM) throughput % | 14.166549 % |
| Achieved occupancy % | 12.299352 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

