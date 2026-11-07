# Removed experiments (R1–R3)

Three rejected optimizations of v3, taken out of the main profiling log because they are not used in the report.
They are labelled R1–R3 (they were measured as Runs 18–20, before Run 18 was reassigned to v8_vlate).
Each folder holds the kernel and the run that measured it. Times: fused forward, N=8192, B=2, H=8, 64x64 tiles
(v3: 0.447 ms at D=64, 0.808 ms at D=128).

| Folder | Label | What it tried | Result |
|---|---|---|---|
| `v8_pipelined/` | R1 | software pipelining: load K/V and compute Q·Kᵀ of tile j+1 before the softmax and P·V of tile j | 15% fewer stall cycles but 18% more instructions; 0.452 ms (D=64), 1.164 ms (D=128, 44% slower) |
| `v9_lazyrescale/` | R2 | rescale acc and l only when a row max grows by more than 8 (log2 units) | data-dependent branch; 0.471 ms (D=64), 0.826 ms (D=128) |
| `v10_exp2fp16/` | R3 | exp2 evaluated in FP16 | 66% more instructions; 0.475 ms (D=64), 1.134 ms (D=128, 40% slower) |

`RUNS_R1-R3.md` and `runs_R1-R3.csv` are their log sections and raw timings.
The kernels were measured at `profiling/kernels/v8_pipelined.py` etc. (the path in the logs).
The names v8–v10 here are earlier experiments; the current `v8` is `../../2_rejected/v8_vlate/`.
