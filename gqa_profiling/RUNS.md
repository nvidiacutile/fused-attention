# Performance runs

One section per run, newest at the bottom. Times are IST. Changes compare the same size and tile on the same GPU.

## Run 1 — 2026-11-05 00:55 IST

**Note:** gqa/v0: baseline (H_kv=2), no optimizations

### Environment

| Environment | |
|---|---|
| Kernel file | attention/gqa/fused_kernels/v0_baseline.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-563720 |
| Git commit | 40a1914 |
| Batch B | 2 |
| Heads H | 8 |
| K/V heads H_kv | 2 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run on this GPU | vs baseline on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 29.2 | 0.023 | 1.23× faster | - | - |
| 64 | 2048 | 64x64 | 0.044 | 192.8 | 0.084 | 1.90× faster | - | - |
| 64 | 8192 | 64x64 | 0.447 | 307.8 | 0.551 | 1.23× faster | - | - |
| 128 | 512 | 64x64 | 0.022 | 47.6 | 0.028 | 1.26× faster | - | - |
| 128 | 2048 | 64x64 | 0.074 | 232.5 | 0.116 | 1.57× faster | - | - |
| 128 | 8192 | 64x64 | 0.803 | 342.3 | 1.024 | 1.28× faster | - | - |

_First run on this GPU: it is the baseline for later runs._

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `gqa_profiling/1_optimizations/v0_baseline/run_001/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 444.576000 us |
| Compute (SM) throughput % | 75.368357 % |
| Achieved occupancy % | 24.789574 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

## Run 2 — 2026-11-05 02:35 IST

**Note:** v1: latency hint on K/V loads

### Change

Added latency=4 hint on K/V ct.load

Code diff vs Run 1 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 1: attention/gqa/fused_kernels/v0_baseline.py
+++ Run 2: attention/gqa/fused_kernels/v1_latency.py
@@ -1,3 +1,3 @@
-# v3: v2 + occupancy=2 hint (target 2 blocks per SM)
+# v4: v3 + latency hint on K/V loads
 # GQA: K/V have H_kv heads; query head h reads K/V head h // (H // H_kv). Grid and O stay on H query heads.
 import math
@@ -8,4 +8,6 @@
 ConstInt = ct.Constant[int]
 LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
+
+KV_LATENCY = 4
 
 
@@ -36,6 +38,7 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
+                     order=(0, 1, 3, 2), latency=KV_LATENCY).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
+                    latency=KV_LATENCY).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
```

### Environment

| Environment | |
|---|---|
| Kernel file | attention/gqa/fused_kernels/v1_latency.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-563768 |
| Git commit | af2dd5f (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |
| K/V heads H_kv | 2 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 1) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.1 | 0.024 | 1.26× faster | 3.8% slower | 3.8% slower |
| 64 | 2048 | 64x64 | 0.046 | 188.4 | 0.086 | 1.89× faster | 2.5% slower | 2.5% slower |
| 64 | 8192 | 64x64 | 0.446 | 308.2 | 0.548 | 1.23× faster | 0.1% faster | 0.1% faster |
| 128 | 512 | 64x64 | 0.024 | 45.2 | 0.030 | 1.26× faster | 5.8% slower | 5.8% slower |
| 128 | 2048 | 64x64 | 0.075 | 228.9 | 0.117 | 1.56× faster | 1.6% slower | 1.6% slower |
| 128 | 8192 | 64x64 | 0.802 | 342.8 | 1.027 | 1.28× faster | 0.1% faster | 0.1% faster |

### Nsight Compute

Profiling failed: ERR_NVGPUCTRPERM (no permission for GPU counters).

---

## Run 3 — 2026-11-05 02:37 IST

**Note:** v2: GQA head packing

### Change

One block handles all query heads sharing a K/V head (rows = QUERY_GROUP*TILE_M)

Code diff vs Run 2 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 2: attention/gqa/fused_kernels/v1_latency.py
+++ Run 3: attention/gqa/fused_kernels/v2_headpacking.py
@@ -1,4 +1,4 @@
-# v4: v3 + latency hint on K/V loads
-# GQA: K/V have H_kv heads; query head h reads K/V head h // (H // H_kv). Grid and O stay on H query heads.
+# v4: v3 + GQA head packing (one block handles all query heads sharing a K/V head)
+# GQA: K/V have H_kv heads; query head h reads K/V head h // (H // H_kv).
 import math
 import numpy as np
@@ -9,24 +9,24 @@
 LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
 
-KV_LATENCY = 4
-
 
 @ct.kernel(occupancy=2)
-def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, QUERY_GROUP: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
-    bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
+def fused_attn_kernel(Q, K, V, O, scale: float, H_KV: ConstInt, QUERY_GROUP: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
+    bkv = ct.bid(0)
     i = ct.num_blocks(1) - 1 - ct.bid(1)    # which Q tile; heaviest (last) Q tiles launch first
-    b = bh // H                   # goes up by 1 every H [0 to BATCH-1]
-    h = bh % H                    # varies from 0 to H-1 in every H
-    kv_h = h // QUERY_GROUP          # GQA: K/V head shared by this query head (H_kv = H // QUERY_GROUP)
+    b = bkv // H_KV
+    kv_h = bkv % H_KV
+
+    ROWS = QUERY_GROUP * TILE_M
 
     # load my Q tile ONCE
-    q = ct.load(Q, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
+    q = ct.load(Q, index=(b, kv_h, i, 0), shape=(1, QUERY_GROUP, TILE_M, TILE_D)).reshape((ROWS, TILE_D))
 
     # running values, one per query row (FP32) (for each Q tile)
-    m = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
-    l = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
-    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)
+    m = ct.full((ROWS, 1), -np.inf, dtype=np.float32)
+    l = ct.full((ROWS, 1), 0.0, dtype=np.float32)
+    acc = ct.full((ROWS, TILE_D), 0.0, dtype=np.float32)
 
-    rows = i * TILE_M + ct.arange(TILE_M, dtype=np.int32)[:, None]  # which query rows does this block/tile handle 
+    pos = ct.full((QUERY_GROUP, TILE_M), 0, dtype=np.int32) + ct.arange(TILE_M, dtype=np.int32)[None, :]
+    rows = i * TILE_M + pos.reshape((ROWS, 1))  # which query rows does this block/tile handle 
 
     # causal bounds in units of K/V tiles (works for TILE_M != TILE_N)
@@ -38,10 +38,9 @@
     for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
-                     order=(0, 1, 3, 2), latency=KV_LATENCY).reshape((TILE_D, TILE_N))
-        v = ct.load(V, index=(b, kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
-                    latency=KV_LATENCY).reshape((TILE_N, TILE_D))
+                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
+        v = ct.load(V, index=(b, kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
 
         # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
-        s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
+        s = ct.full((ROWS, TILE_N), 0.0, dtype=ct.float32)
         s = ct.mma(q, kt,s) 
         
@@ -68,8 +67,8 @@
 
     out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
-    ct.store(O, index=(b, h, i, 0), tile=out.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))
+    ct.store(O, index=(b, kv_h, i, 0), tile=out.astype(O.dtype).reshape((1, QUERY_GROUP, TILE_M, TILE_D)))
 
 
-def fused_gqa_attention(q, k, v, tile_m=64, tile_n=64):
+def fused_gqa_attention(q, k, v, tile_m=32, tile_n=64):
     B, H, N, D = q.shape
     H_kv = k.shape[1]
@@ -82,4 +81,4 @@
     O = torch.empty_like(q)
     scale = LOG2E / math.sqrt(D)      # log2(e) / sqrt(D), folded into one constant
-    ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H, query_group, tile_m, tile_n, D))
+    ct.launch(torch.cuda.current_stream(), (B * H_kv, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H_kv, query_group, tile_m, tile_n, D))
     return O
```

### Environment

| Environment | |
|---|---|
| Kernel file | attention/gqa/fused_kernels/v2_headpacking.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-563768 |
| Git commit | af2dd5f (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |
| K/V heads H_kv | 2 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 2) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 32x64 | 0.026 | 20.7 | 0.024 | 1.08× slower | - | - |
| 64 | 2048 | 32x64 | 0.077 | 111.4 | 0.086 | 1.12× faster | - | - |
| 64 | 8192 | 32x64 | 0.539 | 255.1 | 0.556 | 1.03× faster | - | - |
| 128 | 512 | 32x64 | 0.100 | 10.8 | 0.030 | 3.36× slower | - | - |
| 128 | 2048 | 32x64 | 0.420 | 40.9 | 0.117 | 3.59× slower | - | - |
| 128 | 8192 | 32x64 | 5.030 | 54.6 | 1.017 | 4.94× slower | - | - |

---

## Run 4 — 2026-11-05 02:38 IST

**Note:** v2: head packing, tile_m=16

### Why (from profiling)

Run 3: 32x64 packs 128 rows/block, D=128 up to 6x slower than v1 (suspected register spills)

### Change

No code change; tile_m 32 -> 16 so rows/block = 64, same as v1

Code diff vs Run 3 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
(no code changes)
```

### Environment

| Environment | |
|---|---|
| Kernel file | attention/gqa/fused_kernels/v2_headpacking.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-563768 |
| Git commit | af2dd5f (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |
| K/V heads H_kv | 2 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 3) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 16x64 | 0.020 | 27.2 | 0.024 | 1.21× faster | - | - |
| 64 | 2048 | 16x64 | 0.045 | 192.7 | 0.086 | 1.94× faster | - | - |
| 64 | 8192 | 16x64 | 0.447 | 307.8 | 0.549 | 1.23× faster | - | - |
| 128 | 512 | 16x64 | 0.024 | 44.9 | 0.030 | 1.26× faster | - | - |
| 128 | 2048 | 16x64 | 0.075 | 228.6 | 0.117 | 1.55× faster | - | - |
| 128 | 8192 | 16x64 | 0.803 | 342.2 | 1.028 | 1.28× faster | - | - |

---

## Run 5 — 2026-11-05 02:44 IST

**Note:** v3: head packing 32x64, occupancy=1

### Why (from profiling)

Run 3: 32x64 (128 rows/block) up to 6x slower at D=128, suspected register spills

### Change

occupancy 2 -> 1 in @ct.kernel

Code diff vs Run 4 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 4: attention/gqa/fused_kernels/v2_headpacking.py
+++ Run 5: attention/gqa/fused_kernels/v3_headpacking_occ1.py
@@ -10,5 +10,5 @@
 
 
-@ct.kernel(occupancy=2)
+@ct.kernel(occupancy=1)
 def fused_attn_kernel(Q, K, V, O, scale: float, H_KV: ConstInt, QUERY_GROUP: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     bkv = ct.bid(0)
```

### Environment

| Environment | |
|---|---|
| Kernel file | attention/gqa/fused_kernels/v3_headpacking_occ1.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-563768 |
| Git commit | af2dd5f (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |
| K/V heads H_kv | 2 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 4) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 32x64 | 0.024 | 22.4 | 0.024 | 1.00× slower | - | - |
| 64 | 2048 | 32x64 | 0.060 | 143.3 | 0.086 | 1.44× faster | - | - |
| 64 | 8192 | 32x64 | 0.526 | 261.4 | 0.551 | 1.05× faster | - | - |
| 128 | 512 | 32x64 | 0.088 | 12.2 | 0.030 | 2.94× slower | - | - |
| 128 | 2048 | 32x64 | 0.361 | 47.6 | 0.116 | 3.10× slower | - | - |
| 128 | 8192 | 32x64 | 4.391 | 62.6 | 1.018 | 4.31× slower | - | - |

---

