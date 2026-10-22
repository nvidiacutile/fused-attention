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

```bash
pytest tests/
```
