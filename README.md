# fused-attention

Fused causal attention (FlashAttention-style forward) written in NVIDIA cuTile, with an unfused cuTile baseline and a PyTorch reference.

## Status (through Oct 21)

| Task | What | Owner | File(s) |
|---|---|---|---|
| T1 | Environment setup | Both | `README.md`, `requirements.txt` |
| T2 | PyTorch causal SDPA reference (FP16 in, FP32 intermediates) | Nishitha | `attention/reference.py` |
| T3 | Unfused cuTile baseline (QKᵀ → softmax → PV) | Nishitha | `attention/unfused.py` |
| T4 | `run_attention.py` CLI | Harshith | `run_attention.py`, `attention/validation.py` |
| T5 | Correctness tests | Harshith | `tests/` |

## Environment

GPU: NVIDIA A30 (Ampere, compute capability 8.0, 24 GB) rented from Jarvislabs.
Exact versions of the driver, Python, torch, cuda-tile, nvcc and Nsight Compute are
written to `env.txt` by `setup.sh` on every fresh instance.

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
