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

