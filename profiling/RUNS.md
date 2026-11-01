# Performance runs

One section per run, newest at the bottom. Times are IST. Changes compare the same size and tile on the same GPU.

## Run 1 — 2026-11-01 01:24 IST

**Note:** v0: baseline, no optimizations

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
| Host | jl-vm-558251 |
| Git commit | 2f878a8 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | Fused vs unfused | vs previous run on this GPU | vs baseline on this GPU |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.020 | 27.3 | 0.022 | 1.14× faster | 2.40× faster | - | - |
| 64 | 2048 | 64x64 | 0.077 | 111.5 | 0.084 | 1.10× faster | 4.23× faster | - | - |
| 64 | 8192 | 64x64 | 0.689 | 199.3 | 0.543 | 1.27× slower | 8.68× faster | - | - |
| 128 | 512 | 64x64 | 0.024 | 44.9 | 0.028 | 1.18× faster | 2.16× faster | - | - |
| 128 | 2048 | 64x64 | 0.112 | 153.5 | 0.115 | 1.03× faster | 3.20× faster | - | - |
| 128 | 8192 | 64x64 | 1.082 | 254.0 | 1.006 | 1.08× slower | 5.67× faster | - | - |

_First run on this GPU: it is the baseline for later runs._

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_001/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 689.888000 us |
| Compute (SM) throughput % | 45.558588 % |
| Achieved occupancy % | 12.348800 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

## Run 2 — 2026-11-01 01:46 IST

**Note:** v1: heavy-first block order (re-run; Run 2 measured the unedited copy)

### Why (from profiling)

Run 1 ncu (D=64,N=8192): causal Q tile i loops over a growing number of K/V tiles, so blocks launched last are the heaviest and leave a tail where few SMs are busy.

### Change

Swapped grid axes so (batch,head) is axis 0 and the Q tile is axis 1, and reversed the Q tile index (i = num_blocks(1)-1-bid(1)), so the heaviest Q tiles launch first across all heads.

Code diff vs Run 1 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 1: profiling/kernels/v0_baseline.py
+++ Run 2: profiling/kernels/v1_heavyfirst.py
@@ -1,3 +1,3 @@
-# v0: baseline = attention/fused.py at commit ea1416a (2026-10-31). Frozen snapshot.
+# v1: v0 + heavy-first Q tile order (grid axes swapped, Q tile index reversed)
 import math
 import numpy as np
@@ -10,6 +10,6 @@
 @ct.kernel
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
-    i = ct.bid(0)                 # which Q tile this block owns
-    bh = ct.bid(1)                # which (batch, head)
+    bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
+    i = ct.num_blocks(1) - 1 - ct.bid(1)    # which Q tile; heaviest (last) Q tiles launch first
     b = bh // H                   # goes up by 1 every H [0 to BATCH-1]
     h = bh % H                    # varies from 0 to H-1 in every H
@@ -72,4 +72,4 @@
     O = torch.empty_like(q)
     scale = 1.0 / math.sqrt(D)
-    ct.launch(torch.cuda.current_stream(), (N // tile_m, B * H, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
+    ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
     return O
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v1_heavyfirst.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-558251 |
| Git commit | 437d4c8 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 1) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.1 | 0.024 | 1.26× faster | 3.0% faster | 3.0% faster |
| 64 | 2048 | 64x64 | 0.064 | 134.2 | 0.086 | 1.34× faster | 16.9% faster | 16.9% faster |
| 64 | 8192 | 64x64 | 0.633 | 217.0 | 0.546 | 1.16× slower | 8.1% faster | 8.1% faster |
| 128 | 512 | 64x64 | 0.025 | 42.9 | 0.028 | 1.14× faster | 4.6% slower | 4.6% slower |
| 128 | 2048 | 64x64 | 0.093 | 184.1 | 0.116 | 1.25× faster | 16.6% faster | 16.6% faster |
| 128 | 8192 | 64x64 | 0.993 | 276.8 | 1.012 | 1.02× faster | 8.2% faster | 8.2% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_002/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 632.416000 us |
| Compute (SM) throughput % | 50.042907 % |
| Achieved occupancy % | 12.315086 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

## Run 3 — 2026-11-01 16:29 IST

**Note:** v2: exp2 + folded constants + fast math (FTZ, approx division)

### Why (from profiling)

v1 ncu (D=64,N=8192): 108M non-fused FP32 instructions (est. speedup ~5%); top: FADD s-m_new at line 55 (33.8M), FADD at line 58 (20.1M), FMUL at line 55 (16.9M, the log2(e) multiply inside exp). FP32 pipe 17.8% busy.

### Change

Pass c = log2(e)/sqrt(D) as scale; removed the per-element *scale after the QK mma; row max scaled once per row; p = exp2(s*c - m), alpha = exp2(m - m_new) with flush_to_zero=True; final out = truediv(acc, l) with flush_to_zero and APPROX rounding.

Code diff vs Run 2 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 2: profiling/kernels/v1_heavyfirst.py
+++ Run 3: profiling/kernels/v2_fusedInstructions.py
@@ -1,3 +1,3 @@
-# v1: v0 + heavy-first Q tile order (grid axes swapped, Q tile index reversed)
+# v2: v1 + exp2 with folded constants (log2(e)/sqrt(D) passed as scale) + flush_to_zero on exp2
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
+        alpha = ct.exp2(m - m_new, flush_to_zero=True)
 
         # this tile's numerators: e^(s - m_new)
-        p =  ct.exp(s - m_new)
+        p = ct.exp2(s * scale - m_new, flush_to_zero=True)   
 
         # 6. update the running sum and the accumulator
@@ -61,5 +62,5 @@
         m = m_new
 
-    out = acc / l
+    out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
     ct.store(O, index=(b, h, i, 0), tile=out.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))
 
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
| Git commit | fab2bd4 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 2) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.7 | 0.024 | 1.28× faster | 2.1% faster | 5.1% faster |
| 64 | 2048 | 64x64 | 0.055 | 156.3 | 0.086 | 1.56× faster | 14.2% faster | 28.7% faster |
| 64 | 8192 | 64x64 | 0.557 | 246.8 | 0.552 | 1.01× slower | 12.1% faster | 19.2% faster |
| 128 | 512 | 64x64 | 0.023 | 46.5 | 0.030 | 1.31× faster | 7.6% faster | 3.3% faster |
| 128 | 2048 | 64x64 | 0.087 | 198.5 | 0.118 | 1.36× faster | 7.2% faster | 22.6% faster |
| 128 | 8192 | 64x64 | 0.947 | 290.4 | 1.024 | 1.08× faster | 4.7% faster | 12.5% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_003/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 556.864000 us |
| Compute (SM) throughput % | 57.780877 % |
| Achieved occupancy % | 12.337128 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

