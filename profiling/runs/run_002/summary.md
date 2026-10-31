## Run 2 — 2026-11-01 01:39 IST

**Note:** v1: heavy-first block order

### Why (from profiling)

Run 1 ncu (D=64,N=8192): causal Q tile i loops over a growing number of K/V tiles, so blocks launched last are the heaviest and leave a tail where few SMs are busy (see Workload Distribution in the details file).

### Change

Swapped grid axes so (batch,head) is axis 0 and the Q tile is axis 1, and reversed the Q tile index (i = num_blocks(1)-1-bid(1)), so the heaviest Q tiles launch first across all heads.

Code diff vs Run 1 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
(no code changes)
```

### Environment

| Environment | |
|---|---|
| Kernel file | profiling/kernels/v1_heavyfirst.py |
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
| Git commit | 8aefacb |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 1) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.020 | 27.3 | 0.022 | 1.12× faster | 0.0% faster | 0.0% faster |
| 64 | 2048 | 64x64 | 0.077 | 111.2 | 0.084 | 1.09× faster | 0.3% slower | 0.3% slower |
| 64 | 8192 | 64x64 | 0.691 | 198.8 | 0.547 | 1.26× slower | 0.3% slower | 0.3% slower |
| 128 | 512 | 64x64 | 0.024 | 44.9 | 0.028 | 1.18× faster | 0.0% faster | 0.0% faster |
| 128 | 2048 | 64x64 | 0.112 | 153.4 | 0.115 | 1.03× faster | 0.1% slower | 0.1% slower |
| 128 | 8192 | 64x64 | 1.083 | 253.9 | 1.009 | 1.07× slower | 0.1% slower | 0.1% slower |

### Nsight Compute

Kernel: fused, D=64, N=8192, tile 64x64. Command: `ncu --set full`. Report: `profiling/runs/run_002/fused_d64_n8192_64x64.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 688.928000 us |
| Compute (SM) throughput % | 45.557481 % |
| Achieved occupancy % | 12.339410 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

