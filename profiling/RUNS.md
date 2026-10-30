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

