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

Jarvislabs instances start empty, so run this each time:

```bash
git clone https://github.com/nvidiacutile/fused-attention.git
cd fused-attention
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv`, installs the pinned `requirements.txt`, runs a cuTile
smoke test (`scripts/smoke_test.py`) and records the environment to `env.txt`.
Commit and push your work before pausing the instance.

## Usage

```bash
python run_attention.py --input qkv.npz --output out.npy
```

`qkv.npz` must contain FP16 arrays `q`, `k`, `v` of shape `(B, H, N, D)`. Unsupported inputs exit with a nonzero status and a clear error message.

## Tests

```bash
pytest tests/
```
