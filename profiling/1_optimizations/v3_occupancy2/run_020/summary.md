## Run 20 — 2026-11-07 21:03 IST

**Note:** v3 Nsight at D=64, N=2048

### Why (from profiling)

third profiled configuration

### Change

no code change

Code diff vs Run 19 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
(no code changes)
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-566184 |
| Git commit | e0a42c7 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 19) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.018 | 30.1 | 0.023 | 1.28× faster | 5.3% faster | 9.6% faster |
| 64 | 2048 | 64x64 | 0.044 | 194.4 | 0.085 | 1.93× faster | 1.4% slower | 42.6% faster |
| 64 | 8192 | 64x64 | 0.448 | 306.5 | 0.552 | 1.23× faster | 0.4% slower | 35.0% faster |
| 128 | 512 | 64x64 | 0.022 | 48.0 | 0.029 | 1.29× faster | 2.6% faster | 6.3% faster |
| 128 | 2048 | 64x64 | 0.074 | 232.9 | 0.118 | 1.59× faster | 1.5% faster | 34.0% faster |
| 128 | 8192 | 64x64 | 0.812 | 338.3 | 1.028 | 1.26× faster | 0.7% slower | 24.9% faster |

### Nsight Compute

Kernel: fused, D=64, N=2048, tile 64x64. Command: `ncu --set full`. Report: `profiling/1_optimizations/v3_occupancy2/run_020/fused_d64_n2048_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 39.232000 us |
| Compute (SM) throughput % | 54.141157 % |
| Achieved occupancy % | 23.705901 % |
| Registers / thread | 128 register/thread |
| Occupancy limit (registers, blocks/SM) | 2.000000 block |

---

