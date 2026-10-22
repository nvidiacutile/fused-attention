# fused-attention

Causal scaled dot-product attention in NVIDIA cuTile: a fused FlashAttention-style forward kernel with online softmax, extended to grouped-query attention (GQA/MQA) and a recompute-based backward pass with PyTorch autograd support, tested and benchmarked against an unfused cuTile baseline and PyTorch SDPA.

## Status

| Task | What | Owner | File(s) |
|---|---|---|---|
| T1 | Environment setup | Both | `README.md`, `requirements.txt` |
| T2 | PyTorch causal SDPA reference (FP16 in, FP32 intermediates) | Nishitha | `attention/reference.py` |
| T3 | Unfused cuTile baseline (QKᵀ → softmax → PV) | Nishitha | `attention/unfused.py` |
| T4 | `run_attention.py` CLI | Harshith | `run_attention.py`, `attention/validation.py` |
| T5 | Correctness tests | Harshith | `tests/` |

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA A30 (Ampere, 24 GB), Jarvislabs |
| Compute capability | 8.0 |
| NVIDIA driver | 580.126.20 (supports CUDA 13.0) |
| Python | 3.10.20 |
| PyTorch | 2.14.1+cu130 (CUDA 13.0) |
| cuda-tile (cuTile) | 1.6.0 |
| nvcc (`nvidia-cuda-nvcc`) | 13.4.92 |
| Nsight Compute | 2024.3.2.0 (build 34861637) |

## Setup

Jarvislabs instances start empty, so after logging in to the A30 instance (Jarvislabs dashboard → Instances → **VS Code**, password from the **Password** button) open a terminal and do the steps below each time.

### 1. GitHub access 

The repo is private, so cloning and pushing need a personal access token (PAT).

On each new instance:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global credential.helper store   # remember the token until the instance is wiped
```

Alternative if `gh` is installed (`which gh`): run `gh auth login` → GitHub.com → HTTPS → Login with a web browser, then enter the one-time code at github.com/login/device.

### 2. Clone and set up

```bash
git clone https://github.com/nvidiacutile/fused-attention.git
# Username: nvidiacutile
# Password: paste the token (not the GitHub password)
cd fused-attention
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv`, installs the pinned `requirements.txt`, runs a cuTilesmoke test (`scripts/smoke_test.py`) and records the  environment to `env.txt`.

### 3. Before pausing the instance

```bash
git add -A && git commit -m "..." && git push
```

Anything not pushed is lost when the instance is paused or destroyed. The instance password also changes after pause/resume.

## Usage

```bash
python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile
```

- **Input:** `INPUT.npz` with contiguous float16 arrays `Q` `[B, H, N, D]`, `K` `[B, H, N, D]`, `V` `[B, H, N, Dv]`.
- **Limits:** 1 ≤ B ≤ 16, 1 ≤ H ≤ 32, 1 ≤ N ≤ 8192, D and Dv ∈ {32, 64, 128}, N divisible by `--block-m` and `--block-n` (default 64).
- **Output:** float16 array of shape `[B, H, N, Dv]`.
- **Backends:** `cutile` (fused kernel), `unfused` (baseline), `reference` (PyTorch), `cutedsl` (optional track, not implemented).
- **Exit status:** 0 success, 2 unsupported input (with an error message), 1 runtime failure.

## Tests

### Correctness

```bash
python tests/run_correctness.py --impl unfused                 # or --impl fused
python tests/run_correctness.py --impl fused --block-m 128 --block-n 64
python tests/run_correctness.py --impl fused --filter N512     # subset by case id
```

Runs the chosen kernel and the FP32 PyTorch reference on 60 fixed-seed cases from
`tests/cases.py`:

- D ∈ {32, 64, 128} × N ∈ {512, 4096, 8192} × tiny/large score magnitudes × (B, H) ∈ {(1, 1), (2, 8), (3, 5)}
- plus the maximum shape (B, H) = (16, 32) at N = 512 for each D and magnitude

Tiny scores (σ = 0.03, score std ≈ 1e-3) give a near-uniform softmax; large scores
(σ = 7, max |score| ≈ 240) overflow `exp` in FP32 unless the softmax subtracts the running max.
A case fails on any NaN/Inf, or if any element violates `|out − ref| ≤ 2e-2 + 2e-2·|ref|`.
Prints max absolute/relative error and PASS/FAIL per case, writes
`results/correctness_<impl>.csv`, and exits 0 only if every case passes.

### Unfused and reference

```bash
source .venv/bin/activate
python tests/run_correctness.py --impl reference    # harness self-check: reference vs itself
python tests/run_correctness.py --impl unfused      # unfused cuTile baseline vs reference
```

Each runs all 60 cases and should end with `CORRECT: all 60 cases passed`, writing
`results/correctness_reference.csv` and `results/correctness_unfused.csv`. Add
`--filter N512` (or any part of a case id) to run a subset. The unfused baseline always
uses 64×64 tiles, so `--block-m` / `--block-n` only affect validation for it.
