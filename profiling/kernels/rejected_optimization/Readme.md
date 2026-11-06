# Rejected optimizations

Changes tried on top of v3 (`../v3_occupancy2.py`, the final kernel) that did not make it faster.
Times: fused forward, N = 8192, B = 2, H = 8, 64x64 tiles. v3 takes 0.447 ms (D=64) and 0.808 ms (D=128).

| Version | Run | What it tried | Why rejected |
|---|---|---|---|
| `v4_occupancy3.py` | 5 | `occupancy=3`: caps registers at 80 so 3 blocks fit per SM | spills registers; 5% faster at D=64 but 40% slower at D=128 |
| `v5_occupancy2latency4.py` | 6 | `latency=4` hint on the K and V loads | no measurable effect |
| `v6_occupancy2latency8.py` | 7 | `latency=8` hint on the K and V loads | no measurable effect |
| `v7_ksplit.py` | 17 | two loops: unmasked tiles first, then the masked diagonal tiles | identical to v3; the mask branch costs nothing |
| `v8_vlate.py` | 21 | V loaded just before P·V, as in NVIDIA TileGym | identical to v3 |

Each run's log, Nsight report and code diff are in `../../runs/run_NNN/` and `../../RUNS.md`.
