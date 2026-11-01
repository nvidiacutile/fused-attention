## Run 3 — 2026-11-01 15:54 IST

**Note:** v2: exp2 + folded constants

### Why (from profiling)

v1 ncu (D=64,N=8192): 108M non-fused FP32 instructions (est. speedup ~5%); top: FADD s-m_new at line 55 (33.8M), FADD at line 58 (20.1M), FMUL at line 55 (16.9M, the log2(e) multiply inside exp). FP32 pipe 17.8% busy.

### Change

Pass c = log2(e)/sqrt(D) as scale; removed the per-element *scale after the QK mma; row max scaled once per row; p = exp2(s*c - m) (one FFMA per element) and alpha = exp2(m - m_new).

Code diff vs Run 2 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 2: profiling/kernels/v1_heavyfirst.py
+++ Run 3: profiling/kernels/v2_fusedInstructions.py
@@ -1,3 +1,3 @@
-# v1: v0 + heavy-first Q tile order (grid axes swapped, Q tile index reversed)
+# v2: v1 + exp2 with constants (scale*log2(e) passed as `scale`; one FFMA per score element) and fused instructions
 import math
 import numpy as np
@@ -6,4 +6,5 @@
 
 ConstInt = ct.Constant[int]
+LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
 
 
@@ -38,5 +39,5 @@
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
         s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
-        s = ct.mma(q, kt,s) *scale
+        s = ct.mma(q, kt,s) 
         
 
@@ -47,11 +48,11 @@
 
         # 3. new running max: bigger of old m and this tile's row max
-        m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True))
+        m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True)*scale)
 
         # correction factor for the old totals: e^(m - m_new)
-        alpha = ct.exp(m - m_new)
+        alpha = ct.exp2(m - m_new)
 
         # this tile's numerators: e^(s - m_new)
-        p =  ct.exp(s - m_new)
+        p = ct.exp2(s * scale - m_new)   
 
         # 6. update the running sum and the accumulator
@@ -71,5 +72,5 @@
     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
     O = torch.empty_like(q)
-    scale = 1.0 / math.sqrt(D)
+    scale = LOG2E / math.sqrt(D)      # log2(e) / sqrt(D), folded into one constant
     ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
     return O
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v2_fusedInstructions.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-558730 |
| Git commit | 0b9d92b |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 2) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 30.2 | 0.024 | 1.33× faster | 6.8% faster | 9.6% faster |
| 64 | 2048 | 64x64 | 0.057 | 151.6 | 0.086 | 1.52× faster | 11.4% faster | 26.4% faster |
| 64 | 8192 | 64x64 | 0.552 | 249.0 | 0.547 | 1.01× slower | 12.9% faster | 19.9% faster |
| 128 | 512 | 64x64 | 0.024 | 45.4 | 0.029 | 1.22× faster | 5.6% faster | 1.3% faster |
| 128 | 2048 | 64x64 | 0.089 | 192.2 | 0.117 | 1.31× faster | 4.2% faster | 20.1% faster |
| 128 | 8192 | 64x64 | 0.968 | 284.0 | 1.025 | 1.06× faster | 2.5% faster | 10.6% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_003/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 555.136000 us |
| Compute (SM) throughput % | 58.608927 % |
| Achieved occupancy % | 12.290433 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

