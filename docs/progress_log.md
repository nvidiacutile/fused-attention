# Progress log

Timestamped record of important runs, results and commits. Times are IST (UTC+05:30).
Add new entries at the bottom of the table, one row per event; keep the columns fixed so the log can later be appended to automatically.

| Date | Time (IST) | Task | Event | Result | Commit | Files |
|---|---|---|---|---|---|---|
| 2026-10-22 | | T2 | PyTorch causal attention reference | correctness harness self-check | | `attention/reference.py` |
| 2026-10-22 | | T3 | Unfused cuTile baseline (QKᵀ → softmax → PV) | | | `attention/unfused.py` |
| 2026-10-29 | 16:18 | T6 | Fused kernel correctness, first full run | 60/60 PASS | | `results/correctness_fused.csv` |
| 2026-10-29 | 16:21 | T2 | Reference correctness re-run | 60/60 PASS | | `results/correctness_reference.csv` |
| 2026-10-29 | 16:23 | T6 | Fused kernel pushed | | `967838a` | `attention/fused.py`, `results/correctness_fused.csv` |
| 2026-10-29 | 16:30 | T7 | Forward benchmark, 64×64 tiles, B=2, H=8, L4 | fused 3.1–9.0× faster than unfused; vs SDPA 1.16× faster (D=128, N=512) to 2.25× slower (D=64, N=8192) | | `benchmarks/bench_forward.py`, `results/bench_forward.csv` |
| 2026-10-29 | 16:39 | T7 | Benchmark analysis and plot | summary + latency plot | | `benchmarks/benchmark_analysis.py`, `results/bench_forward_summary.md`, `results/bench_forward.png` |
| 2026-10-29 | 16:49 | — | README: fused kernel, benchmarking, results, environment | | | `README.md`, `docs/progress_log.md` |

## Iteration notes

### 2026-10-29: fused forward v1 (no optimizations)

- Design: FlashAttention-2 style; one block per (Q tile, batch·head); online softmax; causal tile skipping; diagonal-only masking; FP16 in, FP32 accumulation; `tile_m == tile_n` required.
- Correctness: 60/60.
- Performance (L4, 64×64 tiles, B=2, H=8):

| D | N | fused ms | SDPA ms | fused vs SDPA |
|---|---|---|---|---|
| 64 | 512 | 0.055 | 0.051 | 1.08× slower |
| 64 | 2048 | 0.218 | 0.166 | 1.31× slower |
| 64 | 8192 | 2.216 | 0.986 | 2.25× slower |
| 128 | 512 | 0.056 | 0.066 | 1.16× faster |
| 128 | 2048 | 0.234 | 0.193 | 1.21× slower |
| 128 | 8192 | 2.629 | 1.961 | 1.34× slower |

- Next: Nsight Compute profile of fused vs SDPA at D=64, N=8192; then optimizations (scale Q once, exp2, block ordering, tile sweep).
