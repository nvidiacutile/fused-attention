# Profiling

This directory records every optimization step of the fused causal attention kernel (baseline) as a numbered **run**.
Each run says
   1.**why** the change was made (from the previous profile)
   2.**what** changed (with the code diff and line numbers)
   3.**what happened as a result of the change** (timings, comparison with PyTorch SDPA and with earlier runs, and Nsight Compute metrics).

The live kernel used by the tests and `run_attention.py` is `attention/fused.py`. Optimization work happens on
copies in `profiling/kernels/`; `attention/fused.py` is only updated at the end with the best version.

## Contents
 
```
profiling/
├── readme.md            this file
├── track_run.py         runs one measurement: correctness check -> timing -> (optional) Nsight -> records the run
├── __init__.py          makes `python -m profiling.track_run` work
├── RUNS.md              the log: one section per run (why, change + diff, environment, results, Nsight)
├── runs.csv             every measurement of every run, one row per run x size x implementation
├── kernels/             frozen kernel versions, one per optimization step
│   ├── v1_baseline.py                  first series (Runs 1–4)
│   ├── v2_occupancy2.py
│   ├── v3_occupancy2_heavyfirst.py
│   └── <series>/                       a named series started from a snapshot of attention/fused.py
│       ├── v0_baseline.py                  the snapshot (step 0.5)
│       ├── v1_<change>.py                  v0 + one change
│       └── v2_<change>.py                  v1 + one change …
└── runs/run_NNN/        everything about one run (see "Outputs")
```

## Runs so far

| Run | GPU | Kernel | Change | Fused D=64, N=8192 | vs SDPA |
|---|---|---|---|---|---|
| 1 | L4 | v1 | baseline (imported from the first benchmark) | 2.216 ms | 2.25× slower |
| 2 | RTX PRO 6000 | v1 | baseline | 0.703 ms | 1.27× slower |
| 3 | RTX PRO 6000 | v2 | `occupancy=2` (255 -> 128 registers, 2 blocks/SM) | 0.591 ms | 1.08× slower |
| 4 | RTX PRO 6000 | v3 | + heavy-first Q tile order (grid axes swapped) | 0.508 ms | 1.07× faster |

Full tables for all six sizes are in `RUNS.md`. Compare runs only on the same GPU.

## Steps

### 0. Set up the machine (once per new VM)

Nsight Compute needs access to GPU performance counters, which JarvisLabs **templates** block
(`ERR_NVGPUCTRPERM`). Use a JarvisLabs **VM** and run `ncu` with `sudo`.

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
git clone https://github.com/nvidiacutile/fused-attention.git      # username: nvidiacutile, password: token
cd fused-attention
git config --global user.name "nvidiacutile"
git config --global user.email "nvidia.cutile@gmail.com"
git config --global credential.helper store
bash setup.sh
source .venv/bin/activate
```
Once environment is activated and git is cloned 
```bash
sudo git config --global --add safe.directory "$PWD"               # lets the script read the commit under sudo
```

Check: `nvidia-smi` (driver >= 580 for cuTile) and `which ncu`.

### 0.5 (optional) Start a named series from `attention/fused.py`
 
A **series** is a baseline plus every version derived from it, kept together in one folder
`profiling/kernels/<series>/`. The baseline is always `v0_baseline.py`; each later version is `v1_…`, `v2_…`, one change each.
Use this when `attention/fused.py` has changed and you want to start a fresh line of optimizations from it.
The existing `v1_baseline.py`, `v2_…`, `v3_…` are not touched, and `cp -n` never overwrites an existing file.
 
```bash
SERIES=fa2          # short name: letters, digits, - or _ (e.g. fa2, oct31, after-unfused-fix)
 
mkdir -p profiling/kernels/$SERIES
cp -n attention/fused.py profiling/kernels/$SERIES/v0_baseline.py
sed -i "1i # $SERIES/v0: baseline = attention/fused.py at commit $(git rev-parse --short HEAD) ($(date +%F)). Frozen snapshot." profiling/kernels/$SERIES/v0_baseline.py
head -1 profiling/kernels/$SERIES/v0_baseline.py
git status --short profiling/kernels/          # should show only the new folder: ?? profiling/kernels/<series>/
git add profiling/kernels/$SERIES
git commit -m "profiling/kernels/$SERIES: v0 baseline snapshot of attention/fused.py"
git pull --rebase && git push
```

