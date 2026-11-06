# Removed experiments (v8, v9, v10)

Three rejected optimizations of the v3 kernel, taken out of the main profiling log because they are not used
in the report. Kept here for the record. Times: fused forward, N=8192, B=2, H=8, tile 64x64
(v3: 0.447 ms at D=64, 0.808 ms at D=128).

| File here | Originally at | What it tried | Result |
|---|---|---|---|
| `kernels/v8_pipelined.py` | `profiling/kernels/v8_pipelined.py` (later `kernels/rejected_optimization/`) | software pipelining: load K/V and compute Q·Kᵀ of tile j+1 before the softmax and P·V of tile j | 15% fewer stall cycles but 18% more instructions; 0.452 ms (D=64), 1.164 ms (D=128, 44% slower) |
| `kernels/v9_lazyrescale.py` | `profiling/kernels/v9_lazyrescale.py` (later `kernels/rejected_optimization/`) | rescale acc and l only when a row max grows by more than 8 (log2 units) | data-dependent branch; 0.471 ms (D=64), 0.826 ms (D=128) |
| `kernels/v10_exp2fp16.py` | `profiling/kernels/v10_exp2fp16.py` (later `kernels/rejected_optimization/`) | exp2 evaluated in FP16 | 66% more instructions; 0.475 ms (D=64), 1.134 ms (D=128, 40% slower) |
| `runs/run_018/` | `profiling/runs/run_018/` | Run 18: measurement + Nsight report of v8 | |
| `runs/run_019/` | `profiling/runs/run_019/` | Run 19: measurement + Nsight report of v9 | |
| `runs/run_020/` | `profiling/runs/run_020/` | Run 20: measurement + Nsight report of v10 | |
| `RUNS_18-20.md` | sections "Run 18"–"Run 20" of `profiling/RUNS.md` | the run log entries | |
| `runs_18-20.csv` | rows of runs 18–20 in `profiling/runs.csv` | the raw timings | |

Run numbers 18–20 are left unused in the main log, so Run 21 (v11) keeps its number.
