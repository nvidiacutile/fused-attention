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

