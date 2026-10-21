"""Forward correctness vs the FP32 PyTorch reference (Task 5, owner: Harshith).

Parametrize over the implementation under test so the same cases also run
against the fused kernel later.

Cases to cover:
  - D in {64, 128}
  - N equal to one tile and larger than one tile, divisible by every tested tile config
  - tiny and large score magnitudes
  - several batch sizes and head counts
"""

# TODO(T5)
