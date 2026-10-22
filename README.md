# fused-attention

Causal scaled dot-product attention in NVIDIA cuTile on an A30: a fused FlashAttention-style forward kernel with online softmax, extended to grouped-query attention (GQA/MQA) and a recompute-based backward pass with PyTorch autograd support, tested and benchmarked against an unfused cuTile baseline and PyTorch SDPA.

## Status (through Oct 21)

| Task | What | Owner | File(s) |
|---|---|---|---|
| T1 | Environment setup | Both | `README.md`, `requirements.txt` |
| T2 | PyTorch causal SDPA reference (FP16 in, FP32 intermediates) | Nishitha | `attention/reference.py` |
| T3 | Unfused cuTile baseline (QKᵀ → softmax → PV) | Nishitha | `attention/unfused.py` |
| T4 | `run_attention.py` CLI | Harshith | `run_attention.py`, `attention/validation.py` |
| T5 | Correctness tests | Harshith | `tests/` |

## Environment

Verified on Oct 22, 2026 (recorded by `setup.sh` in `env.txt`):

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

Full pinned package list: `requirements.txt`. `setup.sh` rewrites `env.txt` on every
fresh instance; update this table if the versions change.

## Setup (every new instance)

Jarvislabs instances start empty, so after logging in to the A30 instance
(Jarvislabs dashboard → Instances → **VS Code**, password from the **Password**
button) open a terminal and do the steps below each time.

### 1. GitHub access (token created once)

The repo is private, so cloning and pushing need a personal access token (PAT).

Create it once on github.com as `nvidiacutile`:

1. **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**
2. **Repository access:** Only select repositories → `fused-attention`
3. **Permissions → Repository permissions → Contents:** Read and write
4. Set the expiry after Nov 7, generate, and keep the token in a password manager (GitHub shows it only once).

Never commit the token or paste it into chat.

On each new instance:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global credential.helper store   # remember the token until the instance is wiped
```

Alternative if `gh` is installed (`which gh`): run `gh auth login` → GitHub.com →
HTTPS → Login with a web browser, then enter the one-time code at
github.com/login/device.

### 2. Clone and set up

```bash
git clone https://github.com/nvidiacutile/fused-attention.git
# Username: nvidiacutile
# Password: paste the token (not the GitHub password)
cd fused-attention
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv`, installs the pinned `requirements.txt`, runs a cuTile
smoke test (`scripts/smoke_test.py`) and records the environment to `env.txt`.

### 3. Before pausing the instance

```bash
git add -A && git commit -m "..." && git push
```

Anything not pushed is lost when the instance is paused or destroyed. The
instance password also changes after pause/resume.

## Usage

```bash
python run_attention.py --input qkv.npz --output out.npy
```

`qkv.npz` must contain FP16 arrays `q`, `k`, `v` of shape `(B, H, N, D)`. Unsupported inputs exit with a nonzero status and a clear error message.

## Tests

```bash
pytest tests/
```
