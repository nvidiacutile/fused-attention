"""PyTorch reference for causal SDPA (Task 2, owner: Nishitha).

Takes FP16 Q, K, V of shape (B, H, N, D), computes with FP32 intermediates,
and is the ground truth for all kernel correctness tests.
"""

import math
import torch
import torch.nn.functional as F


def reference_attention(q, k, v):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)


def reference_attention_explicit(q, k, v):
    q32, k32, v32 = q.float(), k.float(), v.float()
    N, D = q32.shape[-2], q32.shape[-1]
    S = (q32 @ k32.transpose(-1, -2)) / math.sqrt(D)
    future = torch.triu(torch.ones(N, N, dtype=torch.bool, device=q.device), diagonal=1)
    S = S.masked_fill(future, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return P @ v32


# name imported by run_attention.py and tests/run_correctness.py
causal_attention_reference = reference_attention_explicit
