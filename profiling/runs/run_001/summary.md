## Run 1 — 2026-10-29 16:30 IST

**Note:** fused v1, no optimizations

| Environment | |
|---|---|
| GPU | NVIDIA L4 |
| Git commit | 967838a |
| Batch B | 2 |
| Heads H | 8 |

### Results (fused kernel, median of 30 runs)

| D | N | Tile | Fused ms | Fused TFLOP/s | SDPA ms | Fused vs SDPA | Fused vs unfused | vs previous run | vs first run on this GPU |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 64x64 | 0.055 | 9.7 | 0.051 | 1.08× slower | 4.00× faster | - | - |
| 64 | 2048 | 64x64 | 0.218 | 39.4 | 0.166 | 1.31× slower | 8.98× faster | - | - |
| 64 | 8192 | 64x64 | 2.216 | 62.0 | 0.986 | 2.25× slower | 8.66× faster | - | - |
| 128 | 512 | 64x64 | 0.056 | 19.1 | 0.066 | 1.16× faster | 3.06× faster | - | - |
| 128 | 2048 | 64x64 | 0.234 | 73.6 | 0.193 | 1.21× slower | 6.84× faster | - | - |
| 128 | 8192 | 64x64 | 2.629 | 104.6 | 1.961 | 1.34× slower | 8.09× faster | - | - |

_First run on this GPU: it is the baseline for later runs._

---

