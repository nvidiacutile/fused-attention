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
| Kernel file | profiling/kernels/v3_occupancy2.py |
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

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_004/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 446.880000 us |
| Compute (SM) throughput % | 75.539317 % |
| Achieved occupancy % | 24.790441 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

## Run 5 — 2026-11-01 19:17 IST

**Note:** v4: occupancy=3 (instead of 2)

### Why (from profiling)

v3 ncu (D=64,N=8192): occupancy limited only by registers (128/thread -> 2 blocks/SM; shared memory 24.7 KB allows 3); achieved 24.8%, no eligible warp in 73% of cycles, est. speedup 24%. Tensor pipe 75.5% busy; Math Pipe Throttle ~20% of stalls.

### Change

occupancy=3 instead of 2 in @ct.kernel, so the compiler targets 3 blocks per SM (<=85 registers/thread).

Code diff vs Run 4 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 4: profiling/kernels/v3_occupancy2.py
+++ Run 5: profiling/kernels/v4_occupancy3.py
@@ -1,3 +1,3 @@
-# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
+# v4: v2 + occupancy=3 hint (target 3 blocks per SM)
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel(occupancy=2)
+@ct.kernel(occupancy=3)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v4_occupancy3.py |
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
| Git commit | 57fa9f2 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 4) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 30.4 | 0.023 | 1.28× faster | 5.9% faster | 10.2% faster |
| 64 | 2048 | 64x64 | 0.050 | 171.4 | 0.086 | 1.72× faster | 15.2% slower | 34.9% faster |
| 64 | 8192 | 64x64 | 0.426 | 322.7 | 0.550 | 1.29× faster | 4.7% faster | 38.2% faster |
| 128 | 512 | 64x64 | 0.034 | 31.5 | 0.029 | 1.18× slower | 47.8% slower | 42.3% slower |
| 128 | 2048 | 64x64 | 0.131 | 131.1 | 0.118 | 1.11× slower | 76.9% slower | 17.2% slower |
| 128 | 8192 | 64x64 | 1.131 | 243.0 | 1.023 | 1.11× slower | 40.0% slower | 4.6% slower |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_005/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 422.048000 us |
| Compute (SM) throughput % | 82.437313 % |
| Achieved occupancy % | 36.950664 % |
| Registers / thread | 80 register/thread |
| Occupancy limit (registers, blocks/SM) | 3.000000 block |

---

## Run 6 — 2026-11-01 19:26 IST

**Note:** v5: occupancy=2 + latency=4 on K/V loads

### Why (from profiling)

v3 ncu (D=64,N=8192): waiting for async K/V loads (Sleeping + Long Scoreboard) is ~35% of stall cycles; the compiler cut shared memory to 24.7 KB/block at occupancy=2, leaving room up to ~50 KB/block for deeper prefetching without losing the 2nd block. v4 (occupancy=3) was 5.6% faster but spills (676 KB).

### Change

Added latency=4 to the K and V ct.load calls in the loop (v3 base, occupancy=2), to prefetch further ahead.

Code diff vs Run 5 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 5: profiling/kernels/v4_occupancy3.py
+++ Run 6: profiling/kernels/v5_occupancy2latency4.py
@@ -1,3 +1,3 @@
-# v4: v2 + occupancy=3 hint (target 3 blocks per SM)
+# v5: v3 + latency=4 hint 
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel(occupancy=3)
+@ct.kernel(occupancy=2)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
@@ -34,6 +34,7 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
+                     order=(0, 1, 3, 2), latency=4).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D),
+                    latency=4).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v5_occupancy2latency4.py |
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
| Git commit | a45c6e3 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 5) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.5 | 0.024 | 1.26× faster | 6.8% slower | 4.1% faster |
| 64 | 2048 | 64x64 | 0.044 | 196.7 | 0.086 | 1.96× faster | 12.8% faster | 43.2% faster |
| 64 | 8192 | 64x64 | 0.449 | 306.1 | 0.559 | 1.24× faster | 5.4% slower | 34.9% faster |
| 128 | 512 | 64x64 | 0.023 | 46.7 | 0.029 | 1.24× faster | 32.4% faster | 3.8% faster |
| 128 | 2048 | 64x64 | 0.075 | 229.3 | 0.118 | 1.58× faster | 42.9% faster | 33.1% faster |
| 128 | 8192 | 64x64 | 0.813 | 338.0 | 1.026 | 1.26× faster | 28.1% faster | 24.8% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_006/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 446.784000 us |
| Compute (SM) throughput % | 75.447806 % |
| Achieved occupancy % | 24.771900 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

## Run 7 — 2026-11-01 19:35 IST

**Note:** v6: occupancy=2 + latency=8 on K/V loads

### Why (from profiling)

v5 (latency=4) compiled to the same kernel as v3 (24.68 KB shared memory, 128 registers, identical instruction count and time). Testing a higher hint to see whether the compiler prefetches deeper; shared memory has room up to ~50 KB/block at 2 blocks/SM.

### Change

latency=8 instead of 4 on the K and V ct.load calls (otherwise identical to v5/v3).

Code diff vs Run 6 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 6: profiling/kernels/v5_occupancy2latency4.py
+++ Run 7: profiling/kernels/v6_occupancy2latency8.py
@@ -1,3 +1,3 @@
-# v5: v3 + latency=4 hint 
+# v6: v3 + latency=8 hint on the K/V loads (occupancy=2)
 import math
 import numpy as np
@@ -34,7 +34,7 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2), latency=4).reshape((TILE_D, TILE_N))
+                     order=(0, 1, 3, 2), latency=8).reshape((TILE_D, TILE_N))
         v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D),
-                    latency=4).reshape((TILE_N, TILE_D))
+                    latency=8).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v6_occupancy2latency8.py |
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
| Git commit | ab2ce3f |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 6) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.5 | 0.024 | 1.27× faster | 0.5% faster | 4.6% faster |
| 64 | 2048 | 64x64 | 0.044 | 196.4 | 0.086 | 1.97× faster | 0.0% faster | 43.2% faster |
| 64 | 8192 | 64x64 | 0.441 | 311.9 | 0.553 | 1.26× faster | 1.9% faster | 36.1% faster |
| 128 | 512 | 64x64 | 0.023 | 46.9 | 0.029 | 1.26× faster | 0.4% faster | 4.2% faster |
| 128 | 2048 | 64x64 | 0.075 | 229.5 | 0.118 | 1.58× faster | 0.1% faster | 33.2% faster |
| 128 | 8192 | 64x64 | 0.814 | 337.5 | 1.028 | 1.26× faster | 0.1% slower | 24.7% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_007/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 440.320000 us |
| Compute (SM) throughput % | 76.950784 % |
| Achieved occupancy % | 24.777720 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

## Run 8 — 2026-11-01 19:39 IST

**Note:** v3 (occupancy=2): tile sweep

### Change

Code diff vs Run 7 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 7: profiling/kernels/v6_occupancy2latency8.py
+++ Run 8: profiling/kernels/v3_occupancy2.py
@@ -1,3 +1,3 @@
-# v6: v3 + latency=8 hint on the K/V loads (occupancy=2)
+# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
 import math
 import numpy as np
@@ -34,7 +34,6 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2), latency=8).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D),
-                    latency=8).reshape((TILE_N, TILE_D))
+                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v3_occupancy2.py |
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
| Git commit | ab2ce3f (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 7) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 32x64 | 0.019 | 27.8 | 0.024 | 1.24× faster | - | - |
| 64 | 512 | 64x32 | 0.019 | 28.4 | 0.024 | 1.26× faster | - | - |
| 64 | 512 | 64x64 | 0.019 | 28.6 | 0.024 | 1.27× faster | 0.0% faster | 4.6% faster |
| 64 | 512 | 64x128 | 0.019 | 28.4 | 0.024 | 1.26× faster | - | - |
| 64 | 512 | 128x64 | 0.025 | 21.7 | 0.024 | 1.03× slower | - | - |
| 64 | 512 | 128x128 | 0.048 | 11.3 | 0.024 | 1.99× slower | - | - |
| 64 | 2048 | 32x64 | 0.072 | 119.0 | 0.086 | 1.20× faster | - | - |
| 64 | 2048 | 64x32 | 0.050 | 173.0 | 0.086 | 1.74× faster | - | - |
| 64 | 2048 | 64x64 | 0.044 | 196.9 | 0.086 | 1.98× faster | 0.2% faster | 43.4% faster |
| 64 | 2048 | 64x128 | 0.058 | 148.5 | 0.086 | 1.49× faster | - | - |
| 64 | 2048 | 128x64 | 0.072 | 119.0 | 0.086 | 1.20× faster | - | - |
| 64 | 2048 | 128x128 | 0.172 | 49.8 | 0.086 | 2.00× slower | - | - |
| 64 | 8192 | 32x64 | 0.793 | 173.4 | 0.555 | 1.43× slower | - | - |
| 64 | 8192 | 64x32 | 0.498 | 276.0 | 0.555 | 1.12× faster | - | - |
| 64 | 8192 | 64x64 | 0.449 | 306.2 | 0.555 | 1.24× faster | 1.9% slower | 34.9% faster |
| 64 | 8192 | 64x128 | 0.533 | 257.9 | 0.555 | 1.04× faster | - | - |
| 64 | 8192 | 128x64 | 0.476 | 288.6 | 0.555 | 1.17× faster | - | - |
| 64 | 8192 | 128x128 | 1.709 | 80.4 | 0.555 | 3.08× slower | - | - |
| 128 | 512 | 32x64 | 0.032 | 33.4 | 0.030 | 1.08× slower | - | - |
| 128 | 512 | 64x32 | 0.023 | 46.5 | 0.030 | 1.29× faster | - | - |
| 128 | 512 | 64x64 | 0.023 | 46.5 | 0.030 | 1.29× faster | 0.9% slower | 3.3% faster |
| 128 | 512 | 64x128 | 0.029 | 36.7 | 0.030 | 1.02× faster | - | - |
| 128 | 512 | 128x64 | 0.101 | 10.7 | 0.030 | 3.39× slower | - | - |
| 128 | 512 | 128x128 | 0.115 | 9.4 | 0.030 | 3.86× slower | - | - |
| 128 | 2048 | 32x64 | 0.153 | 112.5 | 0.117 | 1.30× slower | - | - |
| 128 | 2048 | 64x32 | 0.081 | 212.5 | 0.117 | 1.45× faster | - | - |
| 128 | 2048 | 64x64 | 0.074 | 231.3 | 0.117 | 1.58× faster | 0.7% faster | 33.6% faster |
| 128 | 2048 | 64x128 | 0.097 | 177.5 | 0.117 | 1.21× faster | - | - |
| 128 | 2048 | 128x64 | 0.423 | 40.6 | 0.117 | 3.61× slower | - | - |
| 128 | 2048 | 128x128 | 0.606 | 28.4 | 0.117 | 5.17× slower | - | - |
| 128 | 8192 | 32x64 | 1.928 | 142.6 | 1.030 | 1.87× slower | - | - |
| 128 | 8192 | 64x32 | 0.865 | 317.6 | 1.030 | 1.19× faster | - | - |
| 128 | 8192 | 64x64 | 0.772 | 356.0 | 1.030 | 1.33× faster | 5.2% faster | 28.6% faster |
| 128 | 8192 | 64x128 | 0.944 | 291.3 | 1.030 | 1.09× faster | - | - |
| 128 | 8192 | 128x64 | 5.646 | 48.7 | 1.030 | 5.48× slower | - | - |
| 128 | 8192 | 128x128 | 10.993 | 25.0 | 1.030 | 10.67× slower | - | - |

---

## Run 9 — 2026-11-01 19:57 IST

**Note:** v2 (occupancy 1): large-tile check

### Change

Code diff vs Run 8 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 8: profiling/kernels/v3_occupancy2.py
+++ Run 9: profiling/kernels/v2_fusedInstructions.py
@@ -1,3 +1,3 @@
-# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
+# v2: v1 + exp2 with folded constants (log2(e)/sqrt(D) passed as scale) + flush_to_zero on exp2
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel(occupancy=2)
+@ct.kernel
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
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
| Host | jl-vm-558933 |
| Git commit | 4ca8739 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 8) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.6 | 0.024 | 1.28× faster | 0.0% faster | 4.6% faster |
| 64 | 512 | 64x128 | 0.021 | 25.8 | 0.024 | 1.15× faster | 10.1% slower | - |
| 64 | 512 | 128x64 | 0.023 | 23.3 | 0.024 | 1.04× faster | 6.5% faster | - |
| 64 | 512 | 128x128 | 0.041 | 13.0 | 0.024 | 1.72× slower | 13.1% faster | - |
| 64 | 2048 | 64x64 | 0.056 | 153.9 | 0.086 | 1.53× faster | 28.0% slower | 27.5% faster |
| 64 | 2048 | 64x128 | 0.066 | 129.9 | 0.086 | 1.30× faster | 14.2% slower | - |
| 64 | 2048 | 128x64 | 0.058 | 148.5 | 0.086 | 1.48× faster | 19.8% faster | - |
| 64 | 2048 | 128x128 | 0.138 | 62.3 | 0.086 | 1.61× slower | 20.1% faster | - |
| 64 | 8192 | 64x64 | 0.558 | 246.4 | 0.556 | 1.00× slower | 24.3% slower | 19.1% faster |
| 64 | 8192 | 64x128 | 0.725 | 189.5 | 0.556 | 1.30× slower | 36.1% slower | - |
| 64 | 8192 | 128x64 | 0.508 | 270.6 | 0.556 | 1.10× faster | 6.7% slower | - |
| 64 | 8192 | 128x128 | 1.528 | 90.0 | 0.556 | 2.75× slower | 10.6% faster | - |
| 128 | 512 | 64x64 | 0.023 | 46.5 | 0.030 | 1.28× faster | 0.0% faster | 3.3% faster |
| 128 | 512 | 64x128 | 0.023 | 46.7 | 0.030 | 1.29× faster | 21.2% faster | - |
| 128 | 512 | 128x64 | 0.095 | 11.3 | 0.030 | 3.20× slower | 6.2% faster | - |
| 128 | 512 | 128x128 | 0.105 | 10.2 | 0.030 | 3.55× slower | 8.3% faster | - |
| 128 | 2048 | 64x64 | 0.087 | 198.3 | 0.118 | 1.37× faster | 16.6% slower | 22.6% faster |
| 128 | 2048 | 64x128 | 0.086 | 198.9 | 0.118 | 1.37× faster | 10.7% faster | - |
| 128 | 2048 | 128x64 | 0.370 | 46.4 | 0.118 | 3.13× slower | 12.6% faster | - |
| 128 | 2048 | 128x128 | 0.543 | 31.6 | 0.118 | 4.59× slower | 10.4% faster | - |
| 128 | 8192 | 64x64 | 0.950 | 289.3 | 1.026 | 1.08× faster | 23.1% slower | 12.2% faster |
| 128 | 8192 | 64x128 | 0.917 | 299.6 | 1.026 | 1.12× faster | 2.8% faster | - |
| 128 | 8192 | 128x64 | 4.428 | 62.1 | 1.026 | 4.32× slower | 21.6% faster | - |
| 128 | 8192 | 128x128 | 7.426 | 37.0 | 1.026 | 7.24× slower | 32.4% faster | - |

---

## Run 10 — 2026-11-01 19:57 IST

**Note:** v4 (occupancy 3): small-tile check

### Change

Code diff vs Run 9 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 9: profiling/kernels/v2_fusedInstructions.py
+++ Run 10: profiling/kernels/v4_occupancy3.py
@@ -1,3 +1,3 @@
-# v2: v1 + exp2 with folded constants (log2(e)/sqrt(D) passed as scale) + flush_to_zero on exp2
+# v4: v2 + occupancy=3 hint (target 3 blocks per SM)
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel
+@ct.kernel(occupancy=3)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v4_occupancy3.py |
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
| Git commit | 4ca8739 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 9) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 32x64 | 0.020 | 27.5 | 0.024 | 1.22× faster | - | - |
| 64 | 512 | 64x32 | 0.021 | 26.1 | 0.024 | 1.17× faster | - | - |
| 64 | 512 | 64x64 | 0.019 | 28.4 | 0.024 | 1.26× faster | 0.5% slower | 4.1% faster |
| 64 | 2048 | 32x64 | 0.061 | 141.1 | 0.086 | 1.42× faster | - | - |
| 64 | 2048 | 64x32 | 0.056 | 154.1 | 0.086 | 1.55× faster | - | - |
| 64 | 2048 | 64x64 | 0.050 | 172.8 | 0.086 | 1.73× faster | 10.9% faster | 35.5% faster |
| 64 | 8192 | 32x64 | 0.681 | 201.8 | 0.560 | 1.22× slower | - | - |
| 64 | 8192 | 64x32 | 0.458 | 300.2 | 0.560 | 1.22× faster | - | - |
| 64 | 8192 | 64x64 | 0.428 | 320.9 | 0.560 | 1.31× faster | 23.2% faster | 37.9% faster |
| 128 | 512 | 32x64 | 0.040 | 27.2 | 0.030 | 1.31× slower | - | - |
| 128 | 512 | 64x32 | 0.027 | 39.3 | 0.030 | 1.10× faster | - | - |
| 128 | 512 | 64x64 | 0.034 | 31.7 | 0.030 | 1.13× slower | 46.8% slower | 41.8% slower |
| 128 | 2048 | 32x64 | 0.187 | 91.8 | 0.118 | 1.59× slower | - | - |
| 128 | 2048 | 64x32 | 0.097 | 176.9 | 0.118 | 1.21× faster | - | - |
| 128 | 2048 | 64x64 | 0.132 | 130.4 | 0.118 | 1.12× slower | 52.1% slower | 17.7% slower |
| 128 | 8192 | 32x64 | 2.526 | 108.8 | 0.977 | 2.59× slower | - | - |
| 128 | 8192 | 64x32 | 0.844 | 325.7 | 0.977 | 1.16× faster | - | - |
| 128 | 8192 | 64x64 | 1.086 | 253.0 | 0.977 | 1.11× slower | 14.3% slower | 0.4% slower |

---

## Run 11 — 2026-11-01 20:01 IST

**Note:** v3 at 128x64, D=128: spilling check

### Change

Code diff vs Run 10 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 10: profiling/kernels/v4_occupancy3.py
+++ Run 11: profiling/kernels/v3_occupancy2.py
@@ -1,3 +1,3 @@
-# v4: v2 + occupancy=3 hint (target 3 blocks per SM)
+# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
 import math
 import numpy as np
@@ -9,5 +9,5 @@
 
 
-@ct.kernel(occupancy=3)
+@ct.kernel(occupancy=2)
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v3_occupancy2.py |
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
| Git commit | 08a2d41 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 10) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 128 | 8192 | 128x64 | 5.735 | 47.9 | 1.026 | 5.59× slower | - | - |

### Nsight Compute

Kernel: fused, D=128, N=8192, tile 128x64. Command: `ncu --set full`. Report: `profiling/runs/run_011/fused_d128_n8192_128x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 5.222208 ms |
| Compute (SM) throughput % | 15.865002 % |
| Achieved occupancy % | 16.583012 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

