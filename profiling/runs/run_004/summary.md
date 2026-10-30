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

