# Profiling

This folder records every optimization step of the fused causal attention **forward** kernel. Each step is one
**run**, measured on an NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability 12.0, 188 SMs). A run records

1. **why** the change was made (what the previous Nsight profile showed),
2. **what** changed (the code diff, added automatically), and
3. **what happened** (timings against PyTorch SDPA and earlier runs, plus Nsight Compute metrics).

The final kernel is **v3** (`1_optimizations/v3_occupancy2/`), which is also the live kernel in
`attention/fused/fused.py`.

- [Contents](#contents)
- [Results summary](#results-summary)
- [1. Launch a VM](#1-launch-a-vm)
- [2. Set up the VM](#2-set-up-the-vm)
- [3. Measure a kernel with track_run](#3-measure-a-kernel-with-track_run)
- [4. Add a new version](#4-add-a-new-version)
- [5. Save the run](#5-save-the-run)
- [6. Reproduce an earlier run](#6-reproduce-an-earlier-run)
- [track_run options](#track_run-options)
- [What track_run does](#what-track_run-does)
- [Outputs](#outputs)


## Contents

```
profiling/
├── readme.md            this file
├── RUNS.md              the full log: one section per run (why, change + diff, environment, results, Nsight)
├── runs.csv             every measurement of every run (run x D x N x implementation x tile)
├── track_run.py         one measurement: correctness check -> timing -> (optional) Nsight -> records the run
├── __init__.py          makes `python -m profiling.track_run` work
├── 1_optimizations/     the kept chain v0 -> v3, one change per version
│   └── <version>/       <version>.py  +  run_NNN/
├── 2_rejected/          changes tried on top of v3 that did not help (one line each in its Readme.md)
│   └── <version>/       <version>.py  +  run_NNN/
├── 3_tile_study/        why 64x64: existing versions re-measured at other tile sizes, no code change (Readme.md)
│   └── run_NNN_<what>/
├── tilegym/             Nsight profile of NVIDIA TileGym's cuTile attention, for comparison with v3 (Readme.md)
└── archive/             first profiling series and removed experiments R1–R3; history only (Readme.md)
```

Each kernel version sits next to the run that measured it. Runs are numbered **1–18 in the order they were
measured**. Version numbers (v0–v8) follow the optimization story instead, so the two do not line up: v7 is Run 17
and v8 is Run 18, because the tile-study runs 8–16 were measured in between.

## Results summary

Median fused forward time at N = 8192, B = 2, H = 8, causal, FP16, 64x64 tiles. PyTorch SDPA takes ≈0.55 ms (D=64)
and ≈1.02 ms (D=128).

| Version | Run | Change | D=64 ms | D=128 ms | Outcome |
|---|---|---|---|---|---|
| v0 `v0_baseline` | 1 | FA2-style fused kernel: online softmax, causal tile skipping | 0.689 | 1.082 | baseline |
| v1 `v1_heavyfirst` | 2 | heavy-first block order (the longest query tiles start first) | 0.633 | 0.993 | kept |
| v2 `v2_fusedInstructions` | 3 | exp2 with the scale folded into one constant, plus fast math | 0.557 | 0.947 | kept |
| **v3 `v3_occupancy2`** | 4 | `occupancy=2`: 128 registers/thread, 2 blocks/SM | **0.447** | **0.808** | **kept, final** |
| v4 `v4_occupancy3` | 5 | `occupancy=3` | 0.426 | 1.131 | rejected: spills at D=128 |
| v5 `v5_occupancy2latency4` | 6 | `latency=4` on the K/V loads | 0.449 | 0.813 | rejected: no effect |
| v6 `v6_occupancy2latency8` | 7 | `latency=8` on the K/V loads | 0.441 | 0.814 | rejected: no effect |
| v7 `v7_ksplit` | 17 | unmasked and masked key tiles in two loops | 0.447 | 0.813 | rejected: identical |
| v8 `v8_vlate` | 18 | V loaded just before P·V (TileGym's order) | 0.448 | 0.815 | rejected: identical |

Runs 8–16 are listed in `3_tile_study/Readme.md`. At N = 512 a call takes ≈20 µs, so differences of a few percent
there are noise. Only compare runs measured on the same GPU.

## 1. Launch a VM

Nsight Compute reads GPU performance counters. JarvisLabs **templates** (containers) block them, and `ncu` fails
with `ERR_NVGPUCTRPERM`. Use a JarvisLabs **VM**:

1. On JarvisLabs, create a new instance of type **VM** (not a template) with an **RTX PRO 6000** GPU.
2. Connect to it, using SSH or the browser terminal. You are user `ubuntu`, and `sudo` works without a password.
3. When you are done, **pause or delete** the VM. A running VM is billed per hour; a paused VM still pays for storage.

The repository is private, so `git clone` asks for a username and password. Use username `nvidiacutile` and a GitHub
**personal access token** as the password (not the account password).

## 2. Set up the VM

Run this once per new VM:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip     # setup.sh needs python3-venv
git clone https://github.com/nvidiacutile/fused-attention.git      # username: nvidiacutile, password: token
cd fused-attention
git config --global user.name "nvidiacutile"
git config --global user.email "nvidia.cutile@gmail.com"
git config --global credential.helper store                        # remembers the token after the first push
bash setup.sh                                                      # creates .venv and installs the requirements
source .venv/bin/activate
sudo git config --global --add safe.directory "$PWD"               # lets track_run read the commit under sudo
```

Check the setup:

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv   # driver >= 580 for cuTile
which ncu                                                              # Nsight Compute is on the PATH
python -c "import torch, cuda.tile; print(torch.__version__, torch.cuda.get_device_name())"
```

`credential.helper store` saves the token in plain text in `~/.git-credentials`. That is fine on a throwaway VM, but
do not use it on a shared machine.

Every new terminal needs `cd ~/fused-attention && source .venv/bin/activate` before running anything.

## 3. Measure a kernel with track_run

Re-measure the final kernel with an Nsight profile:

```bash
sudo env "PATH=$PATH" python -m profiling.track_run --ncu \
  --kernel profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py \
  --note "v3 re-run" \
  --reason "check that the results reproduce" \
  --change "no code change"
```

- `sudo` is needed because Nsight Compute reads GPU counters. `env "PATH=$PATH"` keeps the virtual environment's
  `python` and `ncu` visible under sudo.
- Without `--ncu` the command only checks correctness and times the kernel. That takes about a minute; with `--ncu`
  it takes a few minutes.
- The run is stored **next to the kernel**, for example in `profiling/1_optimizations/v3_occupancy2/run_019/`.

For a quick try-out at small sizes, add `--Ds 64 --Ns 512`.

## 4. Add a new version

1. **Copy the latest kept version** into the group it belongs to and give it the next version number:

   ```bash
   mkdir -p profiling/2_rejected/v9_idea
   cp profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py profiling/2_rejected/v9_idea/v9_idea.py
   ```

   Make **one change**. The file must define `fused_attention(q, k, v, tile_m, tile_n)`. Start it with a comment
   line saying what changed, e.g. `# v9: v3 + <change>`.

2. **Commit the kernel first**, so the run records the exact code that was measured:

   ```bash
   git add profiling/2_rejected/v9_idea && git commit -m "v9_idea kernel"
   ```

3. **Measure it**, citing the evidence for the change:

   ```bash
   sudo env "PATH=$PATH" python -m profiling.track_run --ncu \
     --kernel profiling/2_rejected/v9_idea/v9_idea.py \
     --note "v9: <change>" \
     --reason "<what the previous Nsight profile showed, with numbers>" \
     --change "<what was changed, in words>"
   ```

4. **Decide where it belongs.** If it is faster, move its folder to `1_optimizations/` with `git mv`, and copy it to
   `attention/fused/fused.py` if it becomes the new final kernel. If not, leave it in `2_rejected/` and add one line
   to `2_rejected/Readme.md`.

**A tile-size check** on an existing version changes no code. Store it in the tile study with `--out-dir`, then
give the folder a descriptive name:

```bash
sudo env "PATH=$PATH" python -m profiling.track_run --ncu --tiles 128x64 --ncu-tile 128x64 \
  --kernel profiling/1_optimizations/v3_occupancy2/v3_occupancy2.py \
  --out-dir profiling/3_tile_study --note "v3 at 128x64"
git mv profiling/3_tile_study/run_019 profiling/3_tile_study/run_019_v3_128x64
```

## 5. Save the run

Files written under `sudo` belong to root, so hand them back to `ubuntu` before committing:

```bash
sudo chown -R ubuntu:ubuntu ~/fused-attention
git add profiling/ && git commit -m "Run 19: v9 <change>"
git pull --rebase && git push
```

## 6. Reproduce an earlier run

Every run records its git commit and kernel path (`runs.csv`, `RUNS.md`, `run_NNN/env.json`).

```bash
# exactly as measured: same kernel and same track_run
git checkout <commit of the run>
sudo env "PATH=$PATH" python -m profiling.track_run --ncu --kernel <kernel path of the run>
git checkout main

# or a version from the current tree
sudo env "PATH=$PATH" python -m profiling.track_run --ncu \
  --kernel profiling/2_rejected/v4_occupancy3/v4_occupancy3.py --note "v4 re-run"
```

Files were reorganised after they were measured. `profiling/kernels/` and `profiling/runs/` became the version
folders, and `v11_vlate.py` was renamed `v8_vlate.py`. The removed experiments were measured as Runs 18–20 and are
now labelled R1–R3. `runs.csv` keeps each kernel path as it was when measured, and
`git show <commit>:<path>` returns that exact code.

## track_run options

| Flag | Default | Meaning |
|---|---|---|
| `--kernel` | `attention/fused/fused.py` | kernel file to measure; must define `fused_attention(q, k, v, tile_m, tile_n)` |
| `--out-dir` | the kernel's folder | where to store the run folder (`profiling/runs/` for the default kernel) |
| `--note` | | short title of the run, shown in its `RUNS.md` heading |
| `--reason` | | why the change was made; cite the previous profile, e.g. `"Run 2 ncu: 255 regs -> 1 block/SM"` |
| `--change` | | what was changed, in words; the code diff is added automatically |
| `--ncu` | off | also profile one launch with Nsight Compute |
| `--ncu-size` | `64,8192` | D,N of the profiled launch |
| `--ncu-tile` | `64x64` | tile of the profiled launch |
| `--ncu-set` | `full` | Nsight section set: `full` (all sections, slowest), `detailed`, `basic` (fastest) |
| `--ncu-args` | | extra `ncu` arguments, e.g. `"--section WarpStateStats"` |
| `--ncu-kernel` | `fused` | regex for the name of the kernel Nsight profiles |
| `--tiles` | `64x64` | tile configs to time, e.g. `64x64,128x128` |
| `--with-unfused` | off | also time the unfused three-kernel baseline |
| `--Ds` | `64,128` | head dimensions to time |
| `--Ns` | `512,2048,8192` | sequence lengths to time |
| `--B` / `--H` | `2` / `8` | batch size / number of heads |
| `--no-check` | off | skip the correctness check (avoid) |
| `--import-csv` | | record an existing benchmark CSV as a run instead of measuring |
| `--timestamp` / `--gpu` / `--commit` | | for `--import-csv`: when, on which GPU and at which commit it was measured |

## What track_run does

1. **Correctness check.** Compares the kernel with an FP32 PyTorch reference for every tile in `--tiles`, at
   D ∈ {64, 128}, N ∈ {512, 2048}, with normal and large scores (σ = 1 and 7). The tolerance is
   `|out − ref| ≤ 2e-2 + 2e-2·|ref|`, and outputs must be finite. If any case fails, nothing is recorded.
2. **Timing.** Times the fused kernel and PyTorch SDPA (plus unfused with `--with-unfused`) at every D, N and tile:
   10 warm-up calls, then the median of 30 calls timed with CUDA events. Inputs are random FP16 (seed 0) and causal.
   TFLOP/s counts the useful causal work, 2·B·H·N²·D.
3. **Nsight Compute** (`--ncu`). Runs
   `ncu --set full -k regex:fused --launch-skip 1 --launch-count 1` on one launch at `--ncu-size` and `--ncu-tile`,
   skipping the first launch so compilation is not profiled. It saves the report, a text export and the console log.
4. **Recording.**
   - The run number is the largest one in `runs.csv`, plus 1.
   - It collects the environment: GPU, driver, PyTorch, CUDA, cuda-tile, Python, Nsight version and git commit.
   - It compares the results with the previous run and with the first run on the same GPU (same size and tile).
   - It diffs the code against the previous run's kernel (`git show <commit>:<path>`).
   - It appends the new rows to `runs.csv` and a new section to `RUNS.md`, and writes the run folder.

## Outputs

| Output | Where | Contents |
|---|---|---|
| Run log | `RUNS.md` (appended) | per run: number, timestamp (IST), note, **Why**, **Change** with diff, environment, results, Nsight metrics |
| All measurements | `runs.csv` (appended) | run, timestamp, GPU, commit, kernel, note, D, N, B, H, impl, tile, median ms, TFLOP/s |
| Run summary | `run_NNN/summary.md` | this run's section of `RUNS.md` |
| Environment | `run_NNN/env.json` | GPU, compute capability, driver, PyTorch/CUDA, cuda-tile, Python, Nsight, commit, Nsight metrics |
| Code change | `run_NNN/change.diff` | diff between the previous run's kernel and this one |
| Nsight report | `run_NNN/fused_d<D>_n<N>_<tile>.ncu-rep` | full report; open it in the Nsight Compute desktop app |
| Nsight as text | `run_NNN/fused_d<D>_n<N>_<tile>_details.txt` | the same report, readable on GitHub |
| Nsight log | `run_NNN/fused_d<D>_n<N>_<tile>_ncu_log.txt` | `ncu` console output, for errors |

The `.ncu-rep` files of Runs 12 and 13 are very large and are excluded in `.gitignore`; their text exports are
included.
