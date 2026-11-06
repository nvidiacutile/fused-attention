# Profiling

Every optimization step of the fused causal attention forward kernel, measured on an NVIDIA RTX PRO 6000
Blackwell Server Edition (sm_120, 188 SMs). Each kernel version sits next to the run that measured it: the
run records **why** the change was made (from the previous Nsight profile), **what** changed (code diff) and
**what happened** (timings vs PyTorch SDPA, Nsight Compute metrics).

The final kernel is **v3** (`1_optimizations/v3_occupancy2/`); it is the live kernel in `attention/fused/fused.py`.

## Contents

```
profiling/
├── readme.md            this file
├── RUNS.md              full log: one section per run (why, change + diff, environment, results, Nsight)
├── runs.csv             every measurement of every run (run x size x implementation x tile)
├── track_run.py         one measurement: correctness check -> timing -> (optional) Nsight -> records the run
├── 1_optimizations/     the kept chain, one change per version
│   └── <version>/       <version>.py  +  run_NNN/  (summary, diff, environment, Nsight report)
├── 2_rejected/          changes tried on top of v3 that did not help (see its Readme.md)
│   └── <version>/       <version>.py  +  run_NNN/
├── 3_tile_study/        why 64x64: existing versions re-measured at other tiles, no code change (Readme.md)
│   └── run_NNN_<what>/
├── tilegym/             Nsight profile of NVIDIA TileGym's cuTile attention, for comparison with v3
└── archive/             first profiling series and removed experiments (history only)
```

## Summary

Median fused forward time, N = 8192, B = 2, H = 8, causal, FP16, 64x64 tiles. SDPA: ≈0.55 ms (D=64), ≈1.02 ms (D=128).

| Version | Run | Change | D=64 ms | D=128 ms | Outcome |
|---|---|---|---|---|---|
| v0 `v0_baseline` | 1 | FA2-style fused kernel: online softmax, causal tile skipping | 0.689 | 1.082 | baseline |
| v1 `v1_heavyfirst` | 2 | heavy-first block order (longest query tiles start first) | 0.633 | 0.993 | kept |
| v2 `v2_fusedInstructions` | 3 | exp2 with the scale folded into one constant, fast math | 0.557 | 0.947 | kept |
| **v3 `v3_occupancy2`** | 4 | `occupancy=2`: 128 registers/thread, 2 blocks/SM | **0.447** | **0.808** | **kept, final** |
| v4 `v4_occupancy3` | 5 | `occupancy=3` | 0.426 | 1.131 | rejected: spills at D=128 |
| v5 `v5_occupancy2latency4` | 6 | `latency=4` on K/V loads | 0.449 | 0.813 | rejected: no effect |
| v6 `v6_occupancy2latency8` | 7 | `latency=8` on K/V loads | 0.441 | 0.814 | rejected: no effect |
| v7 `v7_ksplit` | 17 | unmasked and masked key tiles in two loops | 0.447 | 0.813 | rejected: identical |
| v8 `v8_vlate` | 21 | V loaded just before P·V (TileGym order) | 0.448 | 0.815 | rejected: identical |

Tile-study runs 8–16 are listed in `3_tile_study/Readme.md`. Run numbers 18–20 belong to removed experiments
(`archive/removed/`). At N = 512 a call takes ≈20 µs, so differences of a few percent there are noise.

## Reproducing a run

Each run records its git commit and kernel path (`runs.csv`, `RUNS.md`, `run_NNN/env.json`).

```bash
# exact reproduction: same code and script as measured
git checkout <commit of the run>
sudo env "PATH=$PATH" python -m profiling.track_run --ncu --kernel <kernel path of the run>
git checkout main

# or measure a version from the current tree
sudo env "PATH=$PATH" python -m profiling.track_run --ncu \
  --kernel profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py --note "v3 re-run"
```

Files were reorganised after they were measured (`profiling/kernels/` and `profiling/runs/` became the version
folders; v11_vlate was renamed v8_vlate). `runs.csv` keeps the path each file had when it was measured, and
`git show <commit>:<path>` returns the exact measured code.

### Set up the machine (once per new VM)

Nsight Compute needs GPU performance counters, which JarvisLabs templates block (`ERR_NVGPUCTRPERM`).
Use a JarvisLabs **VM** and run with `sudo`.

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
git clone https://github.com/nvidiacutile/fused-attention.git      # username: nvidiacutile, password: token
cd fused-attention
git config --global user.name "nvidiacutile"
git config --global user.email "nvidia.cutile@gmail.com"
git config --global credential.helper store
bash setup.sh && source .venv/bin/activate
sudo git config --global --add safe.directory "$PWD"     # lets track_run read the commit under sudo
```

### Measure a new version

```bash
mkdir -p profiling/2_rejected/v9_idea
cp profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py profiling/2_rejected/v9_idea/v9_idea.py
# edit it (one change), commit it, then:
sudo env "PATH=$PATH" python -m profiling.track_run --ncu \
  --kernel profiling/2_rejected/v9_idea/v9_idea.py \
  --note "v9: <change>" --reason "<what the previous profile showed>" --change "<what changed>"
# -> profiling/2_rejected/v9_idea/run_022/
sudo chown -R ubuntu:ubuntu ~/fused-attention
git add profiling/ && git commit -m "Run 22: <change>" && git pull --rebase && git push
```

The run is stored in the kernel's folder. Use `--out-dir profiling/3_tile_study` to store it elsewhere (e.g. an
existing version at another tile size), then give the folder a descriptive name.
A kept version goes in `1_optimizations/`, a rejected one in `2_rejected/`. The kernel file must define
`fused_attention(q, k, v, tile_m, tile_n)`.

## What `track_run` does

1. **Correctness check** against the FP32 PyTorch reference (D 64/128 × N 512/2048 × normal and large scores,
   tolerance `|out − ref| <= 2e-2 + 2e-2·|ref|`). If any case fails, nothing is recorded.
2. **Timing** of fused and SDPA (and unfused with `--with-unfused`): 10 warm-up calls, then the median of 30 calls
   timed with CUDA events. Random FP16 inputs, seed 0, causal.
3. **Nsight Compute** (`--ncu`): profiles one fused launch (default D=64, N=8192) with `--set full`.
4. **Recording**: next run number from `runs.csv`; environment; comparison with the previous and first run on the
   same GPU; code diff against the previous run's kernel (`git show <commit>:<path>`); writes the outputs below.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--kernel` | `attention/fused/fused.py` | kernel file to measure |
| `--out-dir` | the kernel's folder | where to store the run folder |
| `--note` / `--reason` / `--change` | | title, why (cite the previous profile), what changed |
| `--ncu` | off | also run Nsight Compute |
| `--ncu-size` / `--ncu-tile` / `--ncu-set` | `64,8192` / `64x64` / `full` | what Nsight profiles |
| `--ncu-args` | | extra `ncu` arguments, e.g. `"--section WarpStateStats"` |
| `--tiles` | `64x64` | tile configs to time, e.g. `64x64,128x128` |
| `--with-unfused` | off | also time the unfused baseline |
| `--Ds` / `--Ns` / `--B` / `--H` | `64,128` / `512,2048,8192` / `2` / `8` | sizes |
| `--no-check` | off | skip the correctness check (avoid) |

## Outputs of one run (`run_NNN/`)

| File | Contents |
|---|---|
| `summary.md` | this run's section of `RUNS.md` |
| `env.json` | GPU, driver, PyTorch/CUDA, cuda-tile, Nsight version, commit, Nsight metrics |
| `change.diff` | code diff against the previous run's kernel |
| `fused_d<D>_n<N>_<tile>.ncu-rep` | full Nsight report (open in the Nsight Compute app) |
| `…_details.txt` / `…_ncu_log.txt` | the report as text (readable on GitHub) / ncu console output |

The `.ncu-rep` files of Runs 12 and 13 are very large and excluded in `.gitignore`; their text exports are included.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERR_NVGPUCTRPERM` | run with `sudo env "PATH=$PATH" ...` on a VM |
| `Git commit: unknown` | `sudo git config --global --add safe.directory "$PWD"` |
| `correctness check FAILED` | the kernel is wrong for some case; fix it before measuring |
| git permission errors after a run | `sudo chown -R ubuntu:ubuntu ~/fused-attention` |
| `No kernels were profiled` | kernel name didn't match `--ncu-kernel` (default `fused`) |
