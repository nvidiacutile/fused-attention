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

## File structure

**Top level**
- `run_attention.py`: command-line tool that reads Q, K, V from a `.npz` file, runs causal attention with the chosen backend and writes the output to `.npy`.
- `setup.sh`: one-command environment setup (creates `.venv`, installs requirements, writes `env.txt`).
- `requirements.txt`: pinned Python packages.
- `env.txt`: GPU, driver, Python, PyTorch and CUDA versions recorded by `setup.sh`.

**`attention/`**: the attention implementations
- `fused.py`: fused FlashAttention-style cuTile forward kernel with online softmax, configurable query/key tile sizes.
- `unfused.py`: tiled unfused cuTile baseline with three separate kernels (QKᵀ scores → softmax → PV); scores and probabilities are stored in GPU memory.
- `reference.py`: PyTorch causal attention reference (FP16 inputs, FP32 intermediates), the ground truth for correctness.
- `validation.py`: shared input checks (shapes, dtype, limits, tile divisibility), used by the CLI, the kernels and the tests.

**`tests/`**: correctness testing
- `cases.py`: the fixed list of 84 test cases with reproducible Q, K, V.
- `run_correctness.py`: runs a chosen implementation on every case, compares it with the reference and writes a PASS/FAIL CSV.

**`benchmarks/`**
- `benchmark_forward.py`: times fused, unfused and PyTorch SDPA across head sizes and sequence lengths, and writes a per-setting CSV, a summary CSV and a plot.

**`results/`**: outputs from the two scripts above
- `correctness/`: correctness CSVs, in one folder per implementation (`reference/`, `unfused/`, and `fused_<version>/<query tile>x<key tile>/`).
- `benchmark/`: benchmark CSVs and plots, in `fused_<version>/<query tile>x<key tile>/`.

**`profiling/`**: a record of each optimization step of the fused kernel, with the reasoning, code changes, timings and Nsight Compute metrics for every run.
