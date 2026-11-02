## Run 16 — 2026-11-02 20:16 IST

**Note:** v0 128x64, D=64 N=8192: does halving D remove spilling?

### Change

Code diff vs Run 15 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
(no code changes)
```

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
| Host | jl-vm-560722 |
| Git commit | d7d74ea (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 15) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.021 | 25.8 | 0.024 | 1.15× faster | 5.1% slower | 5.6% slower |
| 64 | 2048 | 64x64 | 0.080 | 106.9 | 0.086 | 1.07× faster | 1.5% slower | 4.4% slower |
| 64 | 8192 | 64x64 | 0.702 | 195.7 | 0.553 | 1.27× slower | 0.3% slower | 1.9% slower |
| 128 | 512 | 64x64 | 0.025 | 42.8 | 0.030 | 1.21× faster | 5.0% slower | 5.0% slower |
| 128 | 2048 | 64x64 | 0.115 | 149.4 | 0.118 | 1.03× faster | 0.8% slower | 2.8% slower |
| 128 | 8192 | 64x64 | 1.100 | 250.0 | 1.027 | 1.07× slower | 0.4% slower | 1.6% slower |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 128x64. Command: `ncu --set full`. Report: `profiling/runs/run_016/fused_d64_n8192_128x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 786.816000 us |
| Compute (SM) throughput % | 40.535271 % |
| Achieved occupancy % | 12.407461 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

