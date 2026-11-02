## Run 16 — 2026-11-02 20:38 IST

**Note:** v0 at 128x128 (large tile), Nsight at D=64 N=8192; compare with Run 1 (64x64)

### Why (from profiling)

Slowest tile in the v0 benchmark: 2.6x slower than 64x64 at D=64 N=8192 (1.796 vs 0.691 ms)

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
| Git commit | 7c5d572 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 15) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 128x128 | 0.050 | 10.7 | 0.023 | 2.14× slower | - | - |
| 64 | 2048 | 128x128 | 0.192 | 44.8 | 0.086 | 2.22× slower | - | - |
| 64 | 8192 | 128x128 | 1.869 | 73.5 | 0.553 | 3.38× slower | - | - |
| 128 | 512 | 128x128 | 0.120 | 8.9 | 0.029 | 4.15× slower | - | - |
| 128 | 2048 | 128x128 | 0.560 | 30.7 | 0.117 | 4.78× slower | - | - |
| 128 | 8192 | 128x128 | 6.434 | 42.7 | 1.027 | 6.26× slower | - | - |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 128x128. Command: `ncu --set full`. Report: `profiling/runs/run_016/fused_d64_n8192_128x128.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 1.801952 ms |
| Compute (SM) throughput % | 17.637306 % |
| Achieved occupancy % | 12.231152 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

