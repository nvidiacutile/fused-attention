# Profiling

This directory records every optimization step of the fused causal attention kernel, starting from the baseline, as a numbered **run**.
Each run records:

1. **Why** the change was made, based on the previous run's profile.
2. **What** changed, with the code diff and line numbers.
3. **What happened** as a result: timings, comparison with PyTorch SDPA and with earlier runs, and Nsight Compute metrics.

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
 
A **series** is a baseline (`v0_baseline.py`) plus its versions (`v1_…`, `v2_…`, one change each) in `profiling/kernels/<series>/`.
Start one when `attention/fused.py` has changed. Existing files are never overwritten.
 
```bash
SERIES= _____         # short name: letters, digits, - or _ (e.g. fa2, oct31, after-unfused-fix)
 
mkdir -p profiling/kernels/$SERIES
cp -n attention/fused.py profiling/kernels/$SERIES/v0_baseline.py
sed -i "1i # $SERIES/v0: baseline = attention/fused.py at commit $(git rev-parse --short HEAD) ($(date +%F)). Frozen snapshot." profiling/kernels/$SERIES/v0_baseline.py
head -1 profiling/kernels/$SERIES/v0_baseline.py
git status --short profiling/kernels/          # should show only the new folder: ?? profiling/kernels/<series>/
git add profiling/kernels/$SERIES
git commit -m "profiling/kernels/$SERIES: v0 baseline snapshot of attention/fused.py"
git pull --rebase && git push
```

| Command | What it does |
|---|---|
| `SERIES=fa2` | sets a shell variable; every `$SERIES` below becomes `fa2`. It lasts only for this terminal session, so set it again after reconnecting |
| `mkdir -p …/$SERIES` | creates the series folder (`-p`: no error if it exists) |
| `cp -n …` | copies the live kernel as `v0_baseline.py`; `-n` never overwrites an existing file |
| `sed -i "1i …"` | inserts a comment as line 1 with the series name, source commit hash and date |
| `head -1` | shows that first line, to check it |
| `git status --short` | `??` = new untracked file/folder; if any existing kernel shows `M` (modified), stop |
| `git add` / `commit` / `pull --rebase` / `push` | saves the snapshot and uploads it (pulling teammates' changes first) |

### 1. Profile the baseline

For a named series (step 0.5):

```bash
sudo env "PATH=$PATH" python -m profiling.track_run --ncu \
  --kernel profiling/kernels/$SERIES/v0_baseline.py \
  --note "$SERIES/v0: baseline, no optimizations"
```

For the first series: `--kernel profiling/kernels/v1_baseline.py --note "v1: baseline, no optimizations"`.
This is the first run on this GPU, so it becomes the baseline that later runs are compared against.

### 2. Make one change in a new kernel file

Copy the **previous version of the same series**, bump the number, name the change, and change **one thing**,
so each run's effect has one cause.

**Option A: edit on GitHub** (no terminal editor; the VM can stay paused while you edit)

1. Copy and push the unchanged copy:
   ```bash
   SERIES
   cp -n profiling/kernels/$SERIES/v0_baseline.py profiling/kernels/$SERIES/v1_<change>.py
   git add profiling/kernels/$SERIES
   git commit -m "$SERIES/v1: copy of v0 (to edit)"
   git pull --rebase && git push
   ```
2. On GitHub, open `profiling/kernels/<series>/v1_<change>.py` → pencil (Edit) → make the change and set line 1 to
   `# fa2/v1: v0 + <change> (Run N).` → **Commit changes** with the message `fa2/v1: <change>`.
3. Back on the VM, get the edited file and check it:
   ```bash
   git pull --rebase
   head -1 profiling/kernels/$SERIES/v1_<change>.py
   python -c "import ast; ast.parse(open('profiling/kernels/$SERIES/v1_<change>.py').read()); print('syntax ok')"
   ```

**Option B: edit on the VM**

```bash
#Series set
cp -n profiling/kernels/$SERIES/v0_baseline.py profiling/kernels/$SERIES/v1_<change>.py
nano profiling/kernels/$SERIES/v1_<change>.py
#   make the change, and set line 1 to:  # fa2/v1: v0 + <change> (Run N).
python -c "import ast; ast.parse(open('profiling/kernels/$SERIES/v1_<change>.py').read()); print('syntax ok')"
git add profiling/kernels/$SERIES
git commit -m "$SERIES/v1: <change>"
git push
```

For the next optimization, repeat steps 2–4 starting from the latest version: copy `v1_<change>.py` to `v2_<next change>.py`, and so on.
Commit **before** measuring, so the run records the commit that was measured.
The kernel file must define `fused_attention(q, k, v, tile_m, tile_n)`.

### 3. Measure and profile it

```bash
sudo env "PATH=$PATH" python -m profiling.track_run --ncu \
  --kernel profiling/kernels/$SERIES/v1_<change>.py \
  --note "$SERIES/v1: <short title>" \
  --reason "<what the previous run's Nsight profile showed, with numbers>" \
  --change "<what you changed, in words>"
```

### 4. Save and stop (whenever after profiling)

```bash
sudo chown -R ubuntu:ubuntu ~/fused-attention     # files written under sudo belong to root
git add profiling/
git commit -m "Run N: <change>" #mention change
git pull --rebase && git push
```

Then pause or delete the VM. Anything not pushed is lost when a VM is deleted.

### 5. At the end of profiling

Copy the best version into the live kernel and run the full test suite:

```bash
cp profiling/kernels/<best>.py attention/fused.py      # then remove the snapshot header line if you like
python tests/run_correctness.py --impl fused           # all cases must pass
git add attention/fused.py && git commit -m "fused: adopt <best> from profiling" && git push
```

## What `track_run` does

1. **Correctness check** of the chosen kernel against the FP32 PyTorch reference
   (D 64/128 × N 512/2048 × normal and large scores, tolerance `|out − ref| <= 2e-2 + 2e-2·|ref|`).
   If any case fails, nothing is recorded.
2. **Timing** of fused, PyTorch SDPA (and unfused with `--with-unfused`) at every size:
   10 warm-up calls, then the median of 30 calls timed with CUDA events. Random FP16 inputs, seed 0, causal.
3. **Nsight Compute** (`--ncu`): profiles one fused launch at one size (default D=64, N=8192) with `--set full`.
4. **Recording**: assigns the next run number, collects the environment, compares with the previous run and the
   first run on the same GPU, adds the code diff against the previous run's kernel, and writes the outputs below.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--kernel` | `attention/fused.py` | kernel file to measure |
| `--note` | | short title of the run (heading in RUNS.md) |
| `--reason` | | why the change was made (cite the previous profile) |
| `--change` | | what was changed, in words (the code diff is added automatically) |
| `--ncu` | off | also run Nsight Compute |
| `--ncu-size` | `64,8192` | D,N to profile with Nsight |
| `--ncu-tile` | `64x64` | tile config to profile with Nsight |
| `--ncu-set` | `full` | Nsight section set: `full` (all sections, slowest), `detailed`, `basic` (fastest) |
| `--ncu-args` | | extra arguments passed to `ncu`, e.g. `"--section WarpStateStats"` |
| `--ncu-kernel` | `fused` | regex for the kernel name Nsight profiles |
| `--tiles` | `64x64` | tile configs to time, e.g. `64x64,128x128` |
| `--with-unfused` | off | also time the unfused baseline |
| `--Ds` / `--Ns` | `64,128` / `512,2048,8192` | head dims / sequence lengths |
| `--B` / `--H` | `2` / `8` | batch size / heads |
| `--no-check` | off | skip the correctness check (avoid) |
| `--import-csv` (+ `--timestamp`, `--gpu`, `--commit`) | | record an existing benchmark CSV as a run without re-running |

## Outputs

| Output | Location | Contents |
|---|---|---|
| Run log | `RUNS.md` (appended) | per run: number + timestamp (IST), note, **Why**, **Change** with diff, environment, results table, Nsight metrics |
| All measurements | `runs.csv` (appended) | run, timestamp, GPU, commit, kernel file, note, D, N, B, H, impl, tile, median ms, TFLOP/s |
| Run summary | `runs/run_NNN/summary.md` | this run's section of RUNS.md |
| Environment | `runs/run_NNN/env.json` | GPU, compute capability, driver, PyTorch/CUDA, cuda-tile, Python, Nsight version, host, commit, extracted Nsight metrics |
| Code change | `runs/run_NNN/change.diff` | diff between the previous run's kernel and this one |
| Measured kernel | `runs/run_NNN/kernel_measured.py` | exact file that was measured |
| Nsight report | `runs/run_NNN/fused_d<D>_n<N>_<tile>.ncu-rep` | full report; open in the Nsight Compute desktop app |
| Nsight as text | `runs/run_NNN/fused_d<D>_n<N>_<tile>_details.txt` | same report, readable on GitHub |
| Nsight log | `runs/run_NNN/fused_d<D>_n<N>_<tile>_ncu_log.txt` | ncu console output, for errors |

## Reading a run

**Results table** (per size):

| Column | Meaning |
|---|---|
| Fused ms | median time of the timed calls (lower is better) |
| Fused TFLOP/s | useful math per second (higher is better) |
| SDPA ms | PyTorch SDPA on the same inputs, measured in the same run |
| Fused vs SDPA | SDPA time ÷ fused time |
| Fused vs unfused | unfused time ÷ fused time (only with `--with-unfused`) |
| vs previous run (Run N) on this GPU | effect of **this** change |
| vs baseline (Run N) on this GPU | total change since the first run on this GPU |

At N = 512 calls take about 20 µs, so differences of a few percent there are measurement noise.

**Nsight table**: a few key metrics; the full report is in `_details.txt`.

**Nsight report** (`_details.txt`), main sections:

| Section | Look at | Tells you |
|---|---|---|
| GPU Speed Of Light | Compute (SM) %, Memory %, DRAM %, Duration | compute-bound, memory-bound, or latency-bound (both low) |
| Launch Statistics | Registers Per Thread, Block Size, Shared Memory, Waves Per SM | resources each block uses |
| Occupancy | Theoretical / Achieved Occupancy, Block Limit Registers / Shared Mem | how many blocks fit per SM and what limits it |
| Scheduler Statistics | No Eligible %, Active / Eligible Warps | how often the GPU has no ready warp |
| Memory Workload | Local Memory Spilling Requests, L2 Hit Rate | register spills; cache reuse |
| Instruction Statistics | fused vs non-fused FP32 instructions | separate multiply + add that could be fused |
| Workload Distribution | SM active cycles above/below average | load imbalance across SMs |

The `OPT` lines give NVIDIA's suggestions with an estimated speedup. The estimates are upper bounds and do not add up;
use them to rank what to try next, then quote the relevant numbers in `--reason`.

## Rules of thumb

- One change per run; commit the kernel before running.
- Compare runs on the same GPU only (the tables do this automatically).
- Keep failed experiments: save the kernel (e.g. `fa2/v3_occupancy3_REJECTED.py`), record the run, then continue from the previous best.
- The kernel file path and the note (`fa2/v1: …`) in RUNS.md show which series and step each run belongs to.
- Push after every run, before pausing or deleting the VM.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERR_NVGPUCTRPERM` | run with `sudo env "PATH=$PATH" ...` on a VM (templates block GPU counters) |
| `Git commit: unknown` in a run | `sudo git config --global --add safe.directory "$PWD"` |
| `correctness check FAILED` | the kernel is wrong for some case; fix it before measuring |
| `git push` rejected | `git pull --rebase`, then `git push` |
| git complains about file permissions after a run | `sudo chown -R ubuntu:ubuntu ~/fused-attention` |
| `No kernels were profiled` | the kernel name didn't match `--ncu-kernel` (default `fused`) |
