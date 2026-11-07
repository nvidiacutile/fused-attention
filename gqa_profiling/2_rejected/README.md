# Rejected optimizations

Changes tried on top of v0 (`../1_optimizations/v0_baseline/`, the current GQA kernel) that did not make it faster.
Each folder holds the kernel and the run(s) that measured it.
Times: fused forward, N = 8192, B = 2, H = 8, H_kv = 2. v0 takes 0.447 ms (D=64) and 0.803 ms (D=128) with 64x64 tiles.

| Version | Run | What it tried | Why rejected |
|---|---|---|---|
| `v1_latency/` | 2 | latency hint on the K and V loads | no measurable effect (0.446 / 0.802 ms) |
| `v2_headpacking/` | 3 | GQA head packing: the query heads sharing a K/V head in one block, 32x64 tiles | slower: 0.539 ms at D=64 (+21%), 5.03 ms at D=128 (6.3x) |
| `v2_headpacking/` | 4 | same kernel, tile_m = 16 (16x64 tiles) | identical to v0 (0.447 / 0.803 ms) |
| `v3_headpacking_occ1/` | 5 | head packing 32x64 with occupancy 2 -> 1 | slower: 0.526 ms at D=64 (+18%), 4.39 ms at D=128 (5.5x) |
