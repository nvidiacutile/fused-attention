# Fused Attention

Causal scaled dot-product attention in NVIDIA cuTile: a fused FlashAttention-style forward kernel with online softmax, extended to grouped-query attention (GQA/MQA) and a recompute-based backward pass with PyTorch autograd support, tested and benchmarked against an unfused cuTile baseline and PyTorch SDPA.

## Setup

**Python version:** the pinned packages in `requirements.txt` (e.g. `filelock==4.2.9`) need Python >= 3.11. If the system `python3` is older (e.g. 3.10 on Ubuntu 22.04), `pip` fails with "No matching distribution found". Point `setup.sh` at a newer Python via the `PYTHON` variable, e.g. with `uv` (step 3).

**1. Check your Python version**

```bash
python3 --version
```

**2. Clone the repo**

```bash
git clone https://github.com/nvidiacutile/fused-attention.git
cd fused-attention
```

**3. Run the setup script**

If `python3` is 3.11 or newer:

```bash
bash setup.sh
```

If it is older, install Python 3.12 with `uv` and pass it to `setup.sh` through `PYTHON` (remove any `.venv` left by an earlier failed run first):

```bash
uv python install 3.12
rm -rf .venv
PYTHON=$(uv python find 3.12) bash setup.sh
```

If `uv` is not installed, get it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

`setup.sh` creates `.venv`, installs the pinned packages from `requirements.txt`, and records the GPU and software versions in `env.txt`.

**4. Activate the virtual environment**

```bash
source .venv/bin/activate
```

## Reproducing the results

The `Makefile` re-runs every correctness check and benchmark behind `results/correctness/` and `results/benchmarks/` and writes them to `results_replicate/`, with the same folder layout and file names, so each file can be compared with its counterpart in `results/`. `results/` itself is not modified, and the folders are created as needed. Run it from the repo root on a CUDA GPU with `.venv` set up and activated (steps 3 and 4). The committed results were measured on an NVIDIA RTX PRO 6000 Blackwell Server Edition.

```bash
make                 # everything: correctness, then benchmarks (61 runs)
make correctness     # results_replicate/correctness/
make benchmarks      # results_replicate/benchmarks/
make list            # print every command without running it
make compare         # after a run: list files present in only one of results/ and results_replicate/
```

- Smaller groups: `correct-reference`, `correct-unfused`, `correct-fused`, `correct-gqa`, `correct-backward`, `bench-fused`, `bench-gqa`, `bench-backward`, `bench-tilegym`.
- Single runs: `make correct-fused/v3_occupancy2/64x128`, `make bench-fused/v0_baseline/32x64`, `make bench-gqa/v0_baseline/64x64/Hkv2`.
- `make -k` keeps going after a failed run (a correctness run with any FAIL exits non-zero).
- `make OUT=<dir>` writes to another folder; `make OUT=results` regenerates the committed results in place.
- `bench-tilegym` needs NVIDIA TileGym installed and fails without it; it runs last.
- Runs are sequential on purpose, since parallel jobs on one GPU would distort the timings. Timings also vary between runs and GPUs, so compare benchmark numbers with a tolerance rather than expecting identical files.

The scripts can also be run on their own from the repo root; without output options they write into `results/`, e.g.:

```bash
python tests/fused/run_correctness.py --impl fused --kernel attention/fused/fused_kernels/v3_occupancy2.py --block-m 64 --block-n 64
python tests/gqa/run_correctness_gqa.py --impl fused --block-m 64 --block-n 64
python -m tests.backward.run_correctness_backward
python benchmarks/benchmark_forward.py --kernel attention/fused/fused_kernels/v3_occupancy2.py --tile-m 64 --tile-n 64
python benchmarks/benchmark_forward_gqa.py --H-kv 2
python -m benchmarks.benchmark_backward
```

## File structure

**Top level**
- `run_attention.py`: command-line tool that reads Q, K, V from a `.npz` file, runs causal attention with the chosen backend and writes the output to `.npy`.
- `run_attention_gqa.py`: same command-line tool for grouped-query attention: K and V have `H_kv` heads (`H_kv` divides `H`; `H_kv == 1` is MQA, `H_kv == H` is plain multi-head), and the cuTile backend is the GQA kernel.
- `setup.sh`: one-command environment setup (creates `.venv`, installs requirements, writes `env.txt`).
- `requirements.txt`: pinned Python packages.
- `Makefile`: re-runs every correctness check and benchmark into `results_replicate/` (see [Reproducing the results](#reproducing-the-results)).
- `env.txt` (written by `setup.sh`, not committed): GPU, driver, Python, PyTorch and CUDA versions recorded by `setup.sh`.

**`attention/`**: the attention implementations
- `fused/fused.py`: fused FlashAttention-style cuTile forward kernel with online softmax, configurable query/key tile sizes.
- `fused/fused_kernels/`: every version of the fused forward kernel from the optimization log (`v0_baseline`, `v1_heavyfirst`, `v2_fusedInstructions`, `v3_occupancy2`); `fused.py` uses v3.
- `gqa/fused_gqa.py`: the fused forward kernel extended to grouped-query attention; query head `h` reads K/V head `h // (H // H_kv)`.
- `gqa/fused_kernels/`: versions of the GQA kernel: `v0_baseline` (the current kernel) and the rejected `v1_latency`, `v2_headpacking`, `v3_headpacking_occ1`.
- `backward/fused_backward.py`: fused forward that also saves the softmax statistics, plus a recompute-based backward (two cuTile kernels: dK/dV and dQ, no atomics), wrapped as a PyTorch autograd function (`fused_attention_autograd`).
- `fused/unfused.py`: tiled unfused cuTile baseline with three separate kernels (QKᵀ scores → softmax → PV); scores and probabilities are stored in GPU memory.
- `gqa/unfused_gqa.py`: the unfused cuTile baseline for GQA/MQA.
- `fused/reference.py`: PyTorch causal attention reference (FP16 inputs, FP32 intermediates), the ground truth for correctness.
- `gqa/reference_gqa.py`: PyTorch reference for GQA/MQA, the ground truth for the GQA tests.
- `fused/validation.py`: shared input checks (shapes, dtype, limits, tile divisibility), used by the CLI, the kernels and the tests.
- `gqa/validation_gqa.py`: the same input checks for GQA/MQA (K and V of shape [B, H_kv, N, D] with H_kv dividing H), used by run_attention_gqa.py and the GQA tests.

**`tests/`**: correctness testing
- `fused/cases.py`: the fixed list of 84 test cases with reproducible Q, K, V.
- `gqa/cases_gqa.py`: the fixed list of 96 GQA test cases (`H_kv` in 1, 2, 4, 8, including MQA and plain MHA) with reproducible Q, K, V.
- `fused/run_correctness.py`: runs a chosen implementation on every case, compares it with the reference and writes a PASS/FAIL CSV.
- `gqa/run_correctness_gqa.py`: the same check over the GQA cases.
- `backward/cases_backward.py`: the fixed list of 7 backward cases (B, H, N, D, sigma) with reproducible Q, K, V and dO, plus the forward+backward timing grid.
- `backward/run_correctness_backward.py`: checks O, dQ, dK and dV of the fused backward against an FP32 PyTorch reference, and times forward+backward against PyTorch SDPA.

**`benchmarks/`**
- `benchmark_forward.py`: times fused, unfused and PyTorch SDPA across head sizes and sequence lengths, and writes a per-setting CSV, a summary CSV and a plot.
- `benchmark_forward_gqa.py`: the same forward benchmark for GQA (fused and unfused GQA kernels vs SDPA with `enable_gqa=True`), run once per `H_kv`.
- `benchmark_backward.py`: times the fused backward (forward, backward and forward+backward, peak memory, TFLOP/s) against naive PyTorch and PyTorch SDPA.
- `benchmark_tilegym.py`: times the fused forward kernel against NVIDIA TileGym's cuTile attention; results in `results/benchmarks/fused/tilegym/`.

**`results/`**: outputs from the correctness and benchmark scripts above
- `correctness/`: correctness CSVs, in one folder per implementation (`reference/`, `unfused/`, and `fused/<version>/<query tile>x<key tile>/`); the GQA runs use `reference_gqa/`, `unfused_gqa/` and `fused_gqa/`.
- `benchmarks/`: benchmark CSVs and plots: `fused/<version>/<query tile>x<key tile>/` (forward, plus the TileGym comparison in `fused/tilegym/`), `fused_gqa/<version>/<query tile>x<key tile>/Hkv<n>/` (GQA forward, one folder per `H_kv`) and `backward/` (backward timings).
- `archive/`: older results kept for reference (the L4 runs, the earlier per-version and TileGym timings, and their logs and comparison scripts).

**`results_replicate/`** (written by `make`): the same `correctness/` and `benchmarks/` layout as `results/`, produced by the `Makefile` for comparison with the committed results.

**`profiling/`**: a record of each optimization step of the fused kernel, with the reasoning, code changes, timings and Nsight Compute metrics for every run.
- `README.md`: how to set up a VM, measure a kernel with `track_run`, add a new version and reproduce a run, plus a results summary.
- `RUNS.md`, `runs.csv`: the full log (one section per run: why, change and diff, environment, results, Nsight) and every measurement of every run.
- `track_run.py`: times a kernel version, optionally profiles it with Nsight Compute (`--ncu`), and records the run.
- `1_optimizations/`: the kept versions `v0_baseline` to `v3_occupancy2` (the final kernel), each with its run folders (summary, environment, Nsight reports).
- `2_rejected/`: changes tried on top of v3 that did not make it faster (`v4_occupancy3` to `v8_vlate`), with why each was rejected.
- `3_tile_study/`: re-measurements at other tile sizes (no code changes) showing why the final kernel uses 64x64 tiles.
- `tilegym/`: Nsight Compute profile of NVIDIA TileGym's cuTile attention (`tile_fmha`), compared with v3.
- `archive/`: history only: the first profiling series (`v0_profiling/`) and experiments taken out of the main log (`removed/`: v8 pipelining, v9 lazy rescaling, v10 FP16 exp2).

**`gqa_profiling/`**: the same record for the GQA/MQA kernel (`attention/gqa/fused_gqa.py`), laid out and used like `profiling/` (`track_run` takes `--H-kv` for the number of K/V heads).
- `1_optimizations/`: the kept version, `v0_baseline` (the current GQA kernel).
- `2_rejected/`: changes tried on top of v0 that did not make it faster (`v1_latency`, `v2_headpacking`, `v3_headpacking_occ1`).
- `RUNS.md`, `runs.csv`, `track_run.py`: the run log, measurements and run tool, as in `profiling/`.
