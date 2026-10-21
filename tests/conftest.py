"""Shared test fixtures (Task 5, owner: Harshith).

- fixed seeds for reproducible inputs
- mixed tolerance: abs_error <= 2e-2 + 2e-2 * abs(reference)
- NaN/Inf check to catch softmax overflow and masking bugs
"""

ATOL = 2e-2
RTOL = 2e-2
SEED = 0

# TODO(T5): seed fixture, make_qkv(B, H, N, D, scale) helper, assert_close(out, ref) helper
