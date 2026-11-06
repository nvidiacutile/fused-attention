# Tile-size study

Why the final kernel uses 64x64 tiles. These runs change no code: existing versions are re-measured or profiled
at other tile sizes. Times: fused forward, N = 8192, B = 2, H = 8.

| Run | Kernel | What it checks | Result |
|---|---|---|---|
| `run_008_v3_tile_sweep/` | v3 | v3 timed at 32x64, 64x32, 64x64, 64x128, 128x64, 128x128 | 64x64 fastest at both D (0.449 ms at D=64, 0.772 ms at D=128) |
| `run_009_v2_large_tiles_occ1/` | v2 | larger tiles, no occupancy cap | 128x64 beats 64x64 at D=64 (0.508 vs 0.558 ms) but is 4.7x slower at D=128 (4.43 vs 0.95 ms) |
| `run_010_v4_small_tiles_occ3/` | v4 | smaller tiles at occupancy 3 | 64x64 fastest at D=64 (0.428 ms); at D=128 every tile is slower than v3 (≥0.84 vs 0.81 ms) |
| `run_011_v3_128x64_d128/` | v3 | Nsight: does 128x64 spill at D=128? | 5.735 ms at D=128 |
| `run_012_v2_128x128_d128/` | v2 | Nsight: 128x128 at D=128 | 7.412 ms (text report only) |
| `run_013_v2_128x64_d128/` | v2 | Nsight: 128x64 at D=128 | 4.512 ms (text report only) |
| `run_014_v2_64x128_d128/` | v2 | Nsight: 64x128 at D=128 | 0.918 ms |
| `run_015_v0_32x64/` | v0 | Nsight: small tile vs Run 1 | +65% instructions for the same tensor-core work |
| `run_016_v0_128x128/` | v0 | Nsight: large tile vs Run 1 | 54.6 M register-spill requests |

Smaller tiles run many more non-matrix instructions per FLOP; tiles with 128 query rows run out of registers
and spill. Runs 8, 15 and 16 are the evidence cited in the report.
