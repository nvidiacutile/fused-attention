"""Unfused cuTile causal attention baseline (Task 3, owner: Nishitha).

FP16 inputs, FP32 accumulation. Three separate kernels:
  1. masked, scaled QKᵀ  -> scores written to global memory
  2. row-wise softmax    -> probabilities written to global memory
  3. P @ V               -> output
"""


def unfused_attention(q, k, v, tile_m: int = 64, tile_n: int = 64):
    # TODO(T3): implement the three cuTile kernels and launch them in sequence
    raise NotImplementedError
