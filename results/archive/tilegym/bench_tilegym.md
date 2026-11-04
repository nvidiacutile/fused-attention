# Our fused kernel vs NVIDIA TileGym FMHA vs PyTorch SDPA

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; B=2, H=8; causal FP16; median of 30 runs. Ours: `profiling/kernels/v3_occupancy2.py` at 64x64.  
Speedup = other_ms / ours_ms (>1 = ours faster).

| D | N | ours ms | TileGym ms | SDPA ms | ours vs TileGym | ours vs SDPA | TileGym vs SDPA |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0123 | 0.0165 | 0.0131 | 1.35x | 1.07x | 0.79x |
| 64 | 512 | 0.0177 | 0.0218 | 0.0238 | 1.23x | 1.34x | 1.09x |
| 64 | 2048 | 0.0443 | 0.0642 | 0.0851 | 1.45x | 1.92x | 1.33x |
| 64 | 8192 | 0.4439 | 0.5117 | 0.5490 | 1.15x | 1.24x | 1.07x |
| 128 | 128 | 0.0153 | 0.0194 | 0.0167 | 1.27x | 1.09x | 0.86x |
| 128 | 512 | 0.0218 | 0.0267 | 0.0290 | 1.22x | 1.33x | 1.09x |
| 128 | 2048 | 0.0739 | 0.1109 | 0.1167 | 1.50x | 1.58x | 1.05x |
| 128 | 8192 | 0.8048 | 0.9707 | 1.0160 | 1.21x | 1.26x | 1.05x |
| **geo. mean** | | | | | **1.29x** | **1.33x** | **1.03x** |
