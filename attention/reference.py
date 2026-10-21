"""PyTorch reference for causal SDPA (Task 2, owner: Nishitha).

Takes FP16 Q, K, V of shape (B, H, N, D), computes with FP32 intermediates,
and is the ground truth for all kernel correctness tests.
"""

import torch


def causal_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # TODO(T2): upcast to FP32, scale by 1/sqrt(D), apply causal mask, softmax, multiply by V
    raise NotImplementedError
