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

