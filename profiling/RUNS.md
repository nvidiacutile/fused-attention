# Performance runs

One section per run, newest at the bottom. Times are IST. Changes compare the same size and tile on the same GPU.

## Run 1 — 2026-10-29 16:30 IST

**Note:** fused v1, no optimizations

| Environment | |
|---|---|
| GPU | NVIDIA L4 |
| Git commit | 967838a |
| Batch B | 2 |
| Heads H | 8 |

### Results (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | Fused vs unfused | vs previous run | vs first run on this GPU |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.055 | 9.7 | 0.051 | 1.08× slower | 4.00× faster | - | - |
| 64 | 2048 | 64x64 | 0.218 | 39.4 | 0.166 | 1.31× slower | 8.98× faster | - | - |
| 64 | 8192 | 64x64 | 2.216 | 62.0 | 0.986 | 2.25× slower | 8.66× faster | - | - |
| 128 | 512 | 64x64 | 0.056 | 19.1 | 0.066 | 1.16× faster | 3.06× faster | - | - |
| 128 | 2048 | 64x64 | 0.234 | 73.6 | 0.193 | 1.21× slower | 6.84× faster | - | - |
| 128 | 8192 | 64x64 | 2.629 | 104.6 | 1.961 | 1.34× slower | 8.09× faster | - | - |

_First run on this GPU: it is the baseline for later runs._

---

## Run 2 — 2026-10-30 19:41 IST

**Note:** fused v1, no optimizations (RTX PRO 6000 VM)

| Environment | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-555946 |
| Git commit | 276b9d0 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Results (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | Fused vs unfused | vs previous run | vs first run on this GPU |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 29.2 | 0.028 | 1.53× faster | 4.32× faster | - | - |
| 64 | 2048 | 64x64 | 0.076 | 112.4 | 0.086 | 1.12× faster | 12.02× faster | - | - |
| 64 | 8192 | 64x64 | 0.703 | 195.6 | 0.553 | 1.27× slower | 21.07× faster | - | - |
| 128 | 512 | 64x64 | 0.024 | 44.0 | 0.029 | 1.21× faster | 3.48× faster | - | - |
| 128 | 2048 | 64x64 | 0.112 | 154.1 | 0.117 | 1.05× faster | 8.63× faster | - | - |
| 128 | 8192 | 64x64 | 1.088 | 252.5 | 1.018 | 1.07× slower | 14.01× faster | - | - |

_First run on this GPU: it is the baseline for later runs._

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Report: `profiling/runs/run_002/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 700.384000 us |
| Compute (SM) throughput % | 45.210624 % |
| Achieved occupancy % | 12.417202 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

## Run 3 — 2026-10-30 20:25 IST

**Note:** occupancy=2 hint

### Why (from profiling)

Run 2 ncu (D=64,N=8192): 255 registers/thread and 32.8 KB shared mem per block limit occupancy to 1 block/SM (12.4% achieved); no warp eligible in 67% of cycles, so the kernel is latency-bound. Est. speedup up to 55%.

### Change

Added occupancy=2 hint to @ct.kernel in attention/fused.py so the compiler targets 2 blocks per SM (at most ~128 registers/thread).

Code diff vs Run 2 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
diff --git a/attention/fused.py b/attention/fused.py
index b022c47..06fa8e9 100644
--- a/attention/fused.py
+++ b/attention/fused.py
@@ -7,5 +7,5 @@ ConstInt = ct.Constant[int]
 
 
-@ct.kernel
+@ct.kernel(occupancy = 2)  #for run 3
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
     i = ct.bid(0)                 # which Q tile this block owns
```

### Environment

| Environment | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-556131 |
| Git commit | d4c9c72 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (2) | vs first run (2) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.1 | 0.024 | 1.24× faster | 3.8% slower | 3.8% slower |
| 64 | 2048 | 64x64 | 0.068 | 126.0 | 0.086 | 1.25× faster | 10.7% faster | 10.7% faster |
| 64 | 8192 | 64x64 | 0.591 | 232.6 | 0.548 | 1.08× slower | 15.9% faster | 15.9% faster |
| 128 | 512 | 64x64 | 0.025 | 42.5 | 0.030 | 1.18× faster | 3.3% slower | 3.3% slower |
| 128 | 2048 | 64x64 | 0.111 | 154.3 | 0.116 | 1.05× faster | 0.2% faster | 0.2% faster |
| 128 | 8192 | 64x64 | 1.024 | 268.5 | 1.015 | 1.01× slower | 5.9% faster | 5.9% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Report: `profiling/runs/run_003/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 609.088000 us |
| Compute (SM) throughput % | 54.281911 % |
| Achieved occupancy % | 23.851923 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

## Run 4 — 2026-10-30 21:00 IST

**Note:** v3: occupancy=2 + heavy-first block order

### Why (from profiling)

Run 3 ncu (D=64,N=8192): SM active cycles vary by about +15%/-13% from the average (est. speedup 12%). Causal attention gives Q tile i a loop of i+1 K/V tiles, and with only 5.45 waves the heavy late tiles launched last leave a tail where few SMs are busy.

### Change

Swapped grid axes so (batch,head) is axis 0 and the Q tile is axis 1, and reversed the Q tile index (i = num_blocks(1)-1-bid(1)), so the heaviest Q tiles launch first across all heads. Kept occupancy=2.

Code diff vs Run 3 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 3: attention/fused.py
+++ Run 4: profiling/kernels/v3_occupancy2_heavyfirst.py
@@ -1,2 +1,3 @@
+#v3: v2 (occupancy=2) + heavy-first Q tile order with swapped grid axes (Run 4). Frozen snapshot.
 import math
 import numpy as np
@@ -9,6 +10,6 @@
 @ct.kernel(occupancy = 2)  #for run 3
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
-    i = ct.bid(0)                 # which Q tile this block owns
-    bh = ct.bid(1)                # which (batch, head)
+    bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
+    i = ct.num_blocks(1) - 1 - ct.bid(1)    # which Q tile; heaviest (last) Q tiles launch first
     b = bh // H
     h = bh % H
@@ -65,4 +66,4 @@
     O = torch.empty_like(q)
     scale = 1.0 / math.sqrt(D)
-    ct.launch(torch.cuda.current_stream(), (N // tile_m, B * H, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
+    ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
     return O
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v3_occupancy2_heavyfirst.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-556131 |
| Git commit | 3ffcf08 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (3) | vs first run (2) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.019 | 28.3 | 0.024 | 1.26× faster | 0.5% faster | 3.3% slower |
| 64 | 2048 | 64x64 | 0.050 | 172.9 | 0.086 | 1.72× faster | 27.1% faster | 34.9% faster |
| 64 | 8192 | 64x64 | 0.508 | 270.3 | 0.543 | 1.07× faster | 13.9% faster | 27.6% faster |
| 128 | 512 | 64x64 | 0.025 | 43.0 | 0.028 | 1.14× faster | 0.8% faster | 2.5% slower |
| 128 | 2048 | 64x64 | 0.083 | 208.1 | 0.116 | 1.41× faster | 25.9% faster | 26.0% faster |
| 128 | 8192 | 64x64 | 0.911 | 301.7 | 1.018 | 1.12× faster | 11.0% faster | 16.3% faster |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Report: `profiling/runs/run_004/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 505.536000 us |
| Compute (SM) throughput % | 66.703265 % |
| Achieved occupancy % | 24.721205 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

## Run 5 — 2026-10-31 18:54 IST

**Note:** tiles/v0: baseline (unequal-tile support), no optimizations

### Change

Code diff vs Run 4 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
--- Run 4: profiling/kernels/v3_occupancy2_heavyfirst.py
+++ Run 5: profiling/kernels/tiles/v0_baseline.py
@@ -1,3 +1,3 @@
-#v3: v2 (occupancy=2) + heavy-first Q tile order with swapped grid axes (Run 4). Frozen snapshot.
+# tiles/v0: baseline = attention/fused.py at commit 95e7266 (2026-10-31). Frozen snapshot.
 import math
 import numpy as np
@@ -8,10 +8,10 @@
 
 
-@ct.kernel(occupancy = 2)  #for run 3
+@ct.kernel
 def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
-    bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
-    i = ct.num_blocks(1) - 1 - ct.bid(1)    # which Q tile; heaviest (last) Q tiles launch first
-    b = bh // H
-    h = bh % H
+    i = ct.bid(0)                 # which Q tile this block owns
+    bh = ct.bid(1)                # which (batch, head)
+    b = bh // H                   # goes up by 1 every H [0 to BATCH-1]
+    h = bh % H                    # varies from 0 to H-1 in every H
 
     # load my Q tile ONCE
@@ -25,5 +25,11 @@
     rows = i * TILE_M + ct.arange(TILE_M, dtype=np.int32)[:, None]  # which query rows does this block/tile handle 
 
-    for j in range(i + 1):        # causal: only K/V tiles 0..i (assumes TILE_M == TILE_N)
+    # causal bounds in units of K/V tiles (works for TILE_M != TILE_N)
+    # last query row of this tile is (i+1)*TILE_M - 1 -> need K/V tiles 0..ceil((i+1)*TILE_M / TILE_N)-1
+    num_kv_tiles = ((i + 1) * TILE_M + TILE_N - 1) // TILE_N
+    # first K/V tile that can contain a col > the first query row (i*TILE_M) -> needs masking
+    mask_start = (i * TILE_M) // TILE_N
+
+    for j in range(num_kv_tiles):
         kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                      order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
@@ -35,6 +41,6 @@
         
 
-        # 2. causal mask on the diagonal tile only
-        if j == i:
+        # 2. causal mask only on tiles that straddle/cross the diagonal
+        if j >= mask_start:
             cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
             s = ct.where(cols > rows, -np.inf, s)
@@ -60,10 +66,10 @@
 
 def fused_attention(q, k, v, tile_m=64, tile_n=64):
-    if tile_m != tile_n:
-        raise NotImplementedError("fused kernel currently requires block_m == block_n")
     B, H, N, D = q.shape
+    if N % tile_m != 0 or N % tile_n != 0:
+        raise ValueError(f"N={N} must be divisible by tile_m={tile_m} and tile_n={tile_n}")
     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
     O = torch.empty_like(q)
     scale = 1.0 / math.sqrt(D)
-    ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
+    ct.launch(torch.cuda.current_stream(), (N // tile_m, B * H, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
     return O
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/tiles/v0_baseline.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-557949 |
| Git commit | d2e94cc (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | Fused vs unfused | vs previous run (Run 4) on this GPU | vs baseline (Run 2) on this GPU |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.021 | 26.2 | 0.024 | 1.17× faster | 2.32× faster | 7.9% slower | 11.4% slower |
| 64 | 2048 | 64x64 | 0.078 | 109.7 | 0.086 | 1.09× faster | 4.19× faster | 57.5% slower | 2.5% slower |
| 64 | 8192 | 64x64 | 0.693 | 198.4 | 0.552 | 1.25× slower | 8.65× faster | 36.3% slower | 1.4% faster |
| 128 | 512 | 64x64 | 0.025 | 42.7 | 0.029 | 1.16× faster | 2.14× faster | 0.4% slower | 2.9% slower |
| 128 | 2048 | 64x64 | 0.113 | 152.0 | 0.116 | 1.03× faster | 3.19× faster | 37.1% slower | 1.4% slower |
| 128 | 8192 | 64x64 | 1.089 | 252.3 | 1.015 | 1.07× slower | 5.63× faster | 19.5% slower | 0.1% slower |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_005/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 692.512000 us |
| Compute (SM) throughput % | 45.583052 % |
| Achieved occupancy % | 12.337018 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

## Run 6 — 2026-10-31 18:59 IST

**Note:** tiles/v0: baseline, no optimizations

### Change

Code diff vs Run 5 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
(no code changes)
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/tiles/v0_baseline.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-557949 |
| Git commit | d2e94cc (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 5) on this GPU | vs baseline (Run 2) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.020 | 27.1 | 0.024 | 1.21× faster | 3.4% faster | 7.6% slower |
| 64 | 2048 | 64x64 | 0.078 | 109.6 | 0.086 | 1.09× faster | 0.1% slower | 2.6% slower |
| 64 | 8192 | 64x64 | 0.695 | 197.9 | 0.554 | 1.25× slower | 0.3% slower | 1.2% faster |
| 128 | 512 | 64x64 | 0.025 | 43.2 | 0.030 | 1.20× faster | 0.8% faster | 2.0% slower |
| 128 | 2048 | 64x64 | 0.113 | 151.7 | 0.117 | 1.03× faster | 0.1% slower | 1.5% slower |
| 128 | 8192 | 64x64 | 1.091 | 252.1 | 1.023 | 1.07× slower | 0.1% slower | 0.2% slower |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_006/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 692.896000 us |
| Compute (SM) throughput % | 45.605168 % |
| Achieved occupancy % | 12.335536 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

