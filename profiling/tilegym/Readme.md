# NVIDIA TileGym comparison profile

Nsight Compute profile of TileGym's cuTile attention (`tile_fmha`, TileGym commit `c343dd3`) at D=64, N=8192,
B=2, H=8, causal FP16, for comparison with our v3 (`../1_optimizations/v3_occupancy2/run_004/`).

| Metric | Ours (v3) | TileGym |
|---|---|---|
| Duration (µs) | 446.9 | 498.1 |
| Executed instructions (M) | 188.7 | 214.7 |
| Registers per thread | 128 | 128 |
| Shared memory per block (KB) | 24.7 | 41.1 |
| Theoretical / achieved occupancy (%) | 33.3 / 24.8 | 33.3 / 23.9 |
| Compute (SM) throughput (%) | 75.5 | 66.9 |
| SM load-imbalance warning | none | est. 13% |

Files: `tilegym_d64_n8192.ncu-rep` (open in Nsight Compute), `tilegym_d64_n8192_details.txt` (text export).
Timing comparison: `results/benchmarks/fused/tilegym/` (from `benchmarks/benchmark_tilegym.py`).

Reproduce (VM, TileGym installed):

    sudo env "PATH=$PATH" ncu --set full --nvtx --nvtx-include "tilegym/" \
      -o profiling/tilegym/tilegym_d64_n8192 python profiling/tilegym/profile_tilegym.py

`profile_tilegym.py` warms up `tile_fmha` (compile + autotune) and wraps one call in an NVTX range named
`tilegym`, so Nsight profiles only that call.
