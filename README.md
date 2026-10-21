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

Fill in after T1:

- GPU model:
- Driver version:
- CUDA Toolkit version:
- Compute capability:
- cuTile version:
- Compilation mode:

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_attention.py --input qkv.npz --output out.npy
```

`qkv.npz` must contain FP16 arrays `q`, `k`, `v` of shape `(B, H, N, D)`. Unsupported inputs exit with a nonzero status and a clear error message.

## Tests

```bash
pytest tests/
```
