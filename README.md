# Fused Attention

Causal scaled dot-product attention in NVIDIA cuTile: a fused FlashAttention-style forward kernel with online softmax, extended to grouped-query attention (GQA/MQA) and a recompute-based backward pass with PyTorch autograd support, tested and benchmarked against an unfused cuTile baseline and PyTorch SDPA.

## Setup

Clone the repo, run the setup script, then activate the virtual environment:

```bash
git clone https://github.com/nvidiacutile/fused-attention.git
cd fused-attention
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv`, installs the pinned packages from `requirements.txt`, and records the GPU and software versions in `env.txt`.

**Python version:** the pinned packages in `requirements.txt` (e.g. `filelock==4.2.9`) need Python >= 3.11. If the system `python3` is older (e.g. 3.10 on Ubuntu 22.04), `pip` fails with "No matching distribution found". Point `setup.sh` at a newer Python via the `PYTHON` variable, e.g. with `uv`:

```bash
uv python install 3.12
rm -rf .venv
PYTHON=$(uv python find 3.12) bash setup.sh
```

## File structure

**Top level**
- `run_attention.py`: command-line tool that reads Q, K, V from a `.npz` file, runs causal attention with the chosen backend and writes the output to `.npy`.
- `run_attention_gqa.py`: same command-line tool for grouped-query attention: K and V have `H_kv` heads (`H_kv` divides `H`; `H_kv == 1` is MQA, `H_kv == H` is plain multi-head), and the cuTile backend is the GQA kernel.
- `setup.sh`: one-command environment setup (creates `.venv`, installs requirements, writes `env.txt`).
- `requirements.txt`: pinned Python packages.
- `env.txt` (written by `setup.sh`, not committed): GPU, driver, Python, PyTorch and CUDA versions recorded by `setup.sh`.

**`attention/`**: the attention implementations
- `fused/fused.py`: fused FlashAttention-style cuTile forward kernel with online softmax, configurable query/key tile sizes.
- `gqa/fused_gqa.py`: the fused forward kernel extended to grouped-query attention; query head `h` reads K/V head `h // (H // H_kv)`.
- `backward/fused_backward.py`: fused forward that also saves the softmax statistics, plus a recompute-based backward (two cuTile kernels: dK/dV and dQ, no atomics), wrapped as a PyTorch autograd function (`fused_attention_autograd`).
- `fused/unfused.py`: tiled unfused cuTile baseline with three separate kernels (QKᵀ scores → softmax → PV); scores and probabilities are stored in GPU memory.
- `fused/reference.py`: PyTorch causal attention reference (FP16 inputs, FP32 intermediates), the ground truth for correctness.
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

**`results/`**: outputs from the correctness and benchmark scripts above
- `correctness/`: correctness CSVs, in one folder per implementation (`reference/`, `unfused/`, and `fused_<version>/<query tile>x<key tile>/`); the GQA runs use `reference_gqa/`, `unfused_gqa/` and `fused_gqa/`.
- `benchmarks/`: benchmark CSVs and plots: `fused/<version>/<query tile>x<key tile>/` (forward, plus the TileGym comparison in `fused/tilegym/`), `fused_gqa/<version>/<query tile>x<key tile>/Hkv<n>/` (GQA forward, one folder per `H_kv`) and `backward/` (backward timings).
- `archive/`: older results kept for reference (the L4 runs, the earlier per-version and TileGym timings, and their logs and comparison scripts).

**`profiling/`**: a record of each optimization step of the fused kernel, with the reasoning, code changes, timings and Nsight Compute metrics for every run.
