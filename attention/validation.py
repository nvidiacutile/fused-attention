"""Input checks shared by run_attention.py, the kernels and the tests (Task 4, owner: Harshith).

Limits come from the project brief:
  Q: [B, H, N, D]   K: [B, H, N, D]   V: [B, H, N, Dv]   all contiguous float16
  1 <= B <= 16, 1 <= H <= 32, 1 <= N <= 8192, D and Dv in {32, 64, 128}
  causal self-attention only, so Nq == Nk == N
  N must be divisible by the query tile (block_m) and key tile (block_n)

Works on NumPy arrays and torch tensors (anything with .shape and .dtype).
Every failure raises ValidationError with a message that names the offending
input and the allowed values, so the CLI can print it and exit nonzero.
"""

MAX_BATCH = 16
MAX_HEADS = 32
MAX_SEQ_LEN = 8192
SUPPORTED_HEAD_DIMS = (32, 64, 128)


class ValidationError(ValueError):
    """Raised for inputs outside the supported scope."""


def _dtype_name(x) -> str:
    # numpy: dtype('float16') -> 'float16'; torch: torch.float16 -> 'float16'
    return str(x.dtype).replace("torch.", "")


def _is_contiguous(x) -> bool:
    if hasattr(x, "is_contiguous"):  # torch
        return x.is_contiguous()
    flags = getattr(x, "flags", None)  # numpy
    return True if flags is None else bool(flags["C_CONTIGUOUS"])


def validate_qkv(q, k, v, block_m: int, block_n: int) -> None:
    """Raise ValidationError if Q, K, V are not supported for the given tile sizes."""
    tensors = {"Q": q, "K": k, "V": v}

    for name, t in tensors.items():
        if _dtype_name(t) != "float16":
            raise ValidationError(f"{name} has dtype {_dtype_name(t)}; only float16 is supported")
        if len(t.shape) != 4:
            raise ValidationError(
                f"{name} must be 4-D [B, H, N, D], got {len(t.shape)}-D shape {tuple(t.shape)}"
            )
        if not _is_contiguous(t):
            raise ValidationError(f"{name} must be contiguous")

    B, H, Nq, D = q.shape
    Bk, Hk, Nk, Dk = k.shape
    Bv, Hv, Nv, Dv = v.shape

    if not (B == Bk == Bv):
        raise ValidationError(f"batch size mismatch: Q={B}, K={Bk}, V={Bv}")
    if not (H == Hk == Hv):
        raise ValidationError(f"head count mismatch: Q={H}, K={Hk}, V={Hv}")
    if not (Nq == Nk == Nv):
        raise ValidationError(
            f"sequence length mismatch: Q={Nq}, K={Nk}, V={Nv}; causal self-attention needs Nq == Nk"
        )
    if Dk != D:
        raise ValidationError(f"K head dim {Dk} must equal Q head dim {D}")

    if not 1 <= B <= MAX_BATCH:
        raise ValidationError(f"batch size B={B} out of range [1, {MAX_BATCH}]")
    if not 1 <= H <= MAX_HEADS:
        raise ValidationError(f"head count H={H} out of range [1, {MAX_HEADS}]")
    if not 1 <= Nq <= MAX_SEQ_LEN:
        raise ValidationError(f"sequence length N={Nq} out of range [1, {MAX_SEQ_LEN}]")
    if D not in SUPPORTED_HEAD_DIMS:
        raise ValidationError(f"head dim D={D} not supported; allowed {SUPPORTED_HEAD_DIMS}")
    if Dv not in SUPPORTED_HEAD_DIMS:
        raise ValidationError(f"value head dim Dv={Dv} not supported; allowed {SUPPORTED_HEAD_DIMS}")

    for label, block in (("query tile block_m", block_m), ("key tile block_n", block_n)):
        if block <= 0:
            raise ValidationError(f"{label}={block} must be positive")
        if Nq % block != 0:
            raise ValidationError(
                f"sequence length N={Nq} is not divisible by the {label}={block}; "
                f"use N that is a multiple of {block} (partial edge tiles are not supported)"
            )
