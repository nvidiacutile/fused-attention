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

