"""End-to-end tests for run_attention.py (Task 5, owner: Harshith).

  - valid .npz in -> .npy out, matches reference
  - nonzero exit status and clear message for:
      non-divisible sequence length, non-FP16 dtype, wrong rank / mismatched shapes,
      unsupported head dimension
"""

# TODO(T5)
