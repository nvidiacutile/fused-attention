"""GQA/MQA input checks shared by run_attention_gqa.py and the GQA tests (MHA-only: attention/fused/validation.py).

Input requirements:
  Q: [B, H, N, D]   K: [B, H_kv, N, D]   V: [B, H_kv, N, D]   all contiguous float16
  1 <= B <= 16, 1 <= H <= 32, 1 <= N <= 8192, D in {32, 64, 128}
  1 <= H_kv <= H and H % H_kv == 0   (H_kv == H is MHA, H_kv == 1 is MQA)
  query head h reads K/V head h // (H // H_kv)
  Q, K and V share the same head dim D (Dv == D)
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

# strips and gets the dtype from numpy and torch
def _dtype_name(x) -> str:
    # numpy: dtype('float16') -> 'float16'; torch: torch.float16 -> 'float16'
    return str(x.dtype).replace("torch.", "")

# checks if data is in row major order in the memory with no gaps 
def _is_contiguous(x) -> bool:
    if hasattr(x, "is_contiguous"):  # torch
        return x.is_contiguous()
    flags = getattr(x, "flags", None)  # numpy
    return True if flags is None else bool(flags["C_CONTIGUOUS"])


def validate_qkv(q, k, v, block_m: int, block_n: int) -> int:
    """Raise ValidationError if Q, K, V are not supported for the given tile sizes.

    Returns the GQA group size H // H_kv (number of query heads per K/V head).
    """
    tensors = {"Q": q, "K": k, "V": v}

    # check dtype
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

    # batch size, head count, sequence length and head dim mismatches
    if not (B == Bk == Bv):
        raise ValidationError(f"batch size mismatch: Q={B}, K={Bk}, V={Bv}")
    # K and V must share one KV head count; Q may have more heads (GQA/MQA)
    if Hk != Hv:
        raise ValidationError(f"K/V head count mismatch: K={Hk}, V={Hv}; K and V must have the same H_kv")
    if not (Nq == Nk == Nv):
        raise ValidationError(
            f"sequence length mismatch: Q={Nq}, K={Nk}, V={Nv}; causal self-attention needs Nq == Nk"
        )
    if not (D == Dk == Dv):
        raise ValidationError(
            f"head dim mismatch: Q={D}, K={Dk}, V={Dv}; Q, K and V must share the same head dim D"
        )

    # check if B, H, N and D are within the specified bounds 
    if not 1 <= B <= MAX_BATCH:
        raise ValidationError(f"batch size B={B} out of range [1, {MAX_BATCH}]")
    if not 1 <= H <= MAX_HEADS:
        raise ValidationError(f"head count H={H} out of range [1, {MAX_HEADS}]")
    H_kv = Hk
    if not 1 <= H_kv <= H:
        raise ValidationError(f"KV head count H_kv={H_kv} out of range [1, H={H}]")
    if H % H_kv != 0:
        raise ValidationError(
            f"query heads H={H} not divisible by KV heads H_kv={H_kv}; "
            f"H_kv must divide H (each group of H // H_kv query heads shares one K/V head)"
        )
    if not 1 <= Nq <= MAX_SEQ_LEN:
        raise ValidationError(f"sequence length N={Nq} out of range [1, {MAX_SEQ_LEN}]")
    if D not in SUPPORTED_HEAD_DIMS:
        raise ValidationError(f"head dim D={D} not supported; allowed {SUPPORTED_HEAD_DIMS}")

    # sequence length tile size divisiblity
    for label, block in (("query tile block_m", block_m), ("key tile block_n", block_n)):
        if block <= 0:
            raise ValidationError(f"{label}={block} must be positive")
        if Nq % block != 0:
            raise ValidationError(
                f"sequence length N={Nq} is not divisible by the {label}={block}; "
                f"use N that is a multiple of {block} (partial edge tiles are not supported)"
            )

    return H // H_kv
