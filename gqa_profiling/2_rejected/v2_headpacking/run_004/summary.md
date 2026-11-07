## Run 4 — 2026-11-05 02:38 IST

**Note:** v2: head packing, tile_m=16

### Why (from profiling)

Run 3: 32x64 packs 128 rows/block, D=128 up to 6x slower than v1 (suspected register spills)

### Change

No code change; tile_m 32 -> 16 so rows/block = 64, same as v1

Code diff vs Run 3 (hunk headers `@@ -old,+new @@` give line numbers):

```diff
(no code changes)
```

### Environment

| Environment | |
|---|---|
| Kernel file | attention/gqa/fused_kernels/v2_headpacking.py |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory (GB) | 95.0 |
| NVIDIA driver | 595.91.07 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0 |
| cuda-tile | 1.6.0 |
| Python | 3.12.3 |
| Nsight Compute | Version 2026.1.1.0 (build 37634170) (public-release) |
| Host | jl-vm-563768 |
| Git commit | af2dd5f (+ uncommitted changes) |
| Batch B | 2 |
| Heads H | 8 |
| K/V heads H_kv | 2 |

### Result (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | vs previous run (Run 3) on this GPU | vs baseline (Run 1) on this GPU |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 16x64 | 0.020 | 27.2 | 0.024 | 1.21× faster | - | - |
| 64 | 2048 | 16x64 | 0.045 | 192.7 | 0.086 | 1.94× faster | - | - |
| 64 | 8192 | 16x64 | 0.447 | 307.8 | 0.549 | 1.23× faster | - | - |
| 128 | 512 | 16x64 | 0.024 | 44.9 | 0.030 | 1.26× faster | - | - |
| 128 | 2048 | 16x64 | 0.075 | 228.6 | 0.117 | 1.55× faster | - | - |
| 128 | 8192 | 16x64 | 0.803 | 342.2 | 1.028 | 1.28× faster | - | - |

---

