## Run 1 — 2026-11-01 01:24 IST

**Note:** v0: baseline, no optimizations

### Environment

| Environment | |
|---|---|
| Kernel file | [profiling/kernels/v0_baseline.py](../../kernels/v0_baseline.py) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-558251 |
| Git commit | 2f878a8 |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | Fused vs unfused | vs previous run on this GPU | vs baseline on this GPU |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.020 | 27.3 | 0.022 | 1.14× faster | 2.40× faster | - | - |
| 64 | 2048 | 64x64 | 0.077 | 111.5 | 0.084 | 1.10× faster | 4.23× faster | - | - |
| 64 | 8192 | 64x64 | 0.689 | 199.3 | 0.543 | 1.27× slower | 8.68× faster | - | - |
| 128 | 512 | 64x64 | 0.024 | 44.9 | 0.028 | 1.18× faster | 2.16× faster | - | - |
| 128 | 2048 | 64x64 | 0.112 | 153.5 | 0.115 | 1.03× faster | 3.20× faster | - | - |
| 128 | 8192 | 64x64 | 1.082 | 254.0 | 1.006 | 1.08× slower | 5.67× faster | - | - |

_First run on this GPU: it is the baseline for later runs._

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_001/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 689.888000 us |
| Compute (SM) throughput % | 45.558588 % |
| Achieved occupancy % | 12.348800 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

