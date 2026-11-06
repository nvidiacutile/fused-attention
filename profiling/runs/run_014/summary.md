## Run 14 — 2026-11-01 20:21 IST

**Note:** v2 (occupancy 1) at 64x128, D=128 N=8192: large-tile profile

### Why (from profiling)

Large tiles were 4-14x slower than 64x64 at D=128 in the tile x occupancy sweep (Runs 8-10); suspected register spilling.

### Change

No code change; tile 64x128.

Code diff vs Run 13 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
(no code changes)
```

### Environment

| Environment | |
|---|---|
| Kernel file | [profiling/kernels/v2_fusedInstructions.py](../../kernels/v2_fusedInstructions.py) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.58.03 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-558933 |
| Git commit | 6b1b160 (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 13) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 128 | 8192 | 64x128 | 0.918 | 299.4 | 1.028 | 1.12× faster | - | - |

### Nsight Compute

Kernel: fused, D=128, N=8192, tile 64x128. Command: `ncu --set full`. Report: `profiling/runs/run_014/fused_d128_n8192_64x128.ncu-rep`

| Metric | Value |
|---|---|
| Duration | 913.792000 us |
| Compute (SM) throughput % | 71.583492 % |
| Achieved occupancy % | 12.424304 % |
| Registers / thread | 255 register/thread |
| Occupancy limit (registers, blocks/SM) | 1.000000 block |

---

