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

