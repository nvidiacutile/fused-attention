"""Shared input checks for the CLI, kernels and tests (Tasks 4 and 5, owner: Harshith).

Every check raises ValueError with a clear message so that run_attention.py can
print it and exit nonzero, and tests can assert on it.
"""

SUPPORTED_HEAD_DIMS = (64, 128)


def validate_qkv(q, k, v, tile_m: int, tile_n: int) -> None:
    """Raise ValueError if Q, K, V are not supported.

    Checks:
      - dtype is FP16 for q, k and v
      - each is 4-D (B, H, N, D) and the three shapes match
      - D is in SUPPORTED_HEAD_DIMS
      - N is divisible by tile_m and tile_n
    """
    # TODO(T4)
    raise NotImplementedError
