"""PyTorch reference for causal SDPA, GQA/MQA version (MHA-only: attention/fused/reference.py).

Takes FP16 Q of shape (B, H, N, D) and K, V of shape (B, H_kv, N, D), computes
with FP32 intermediates, and is the ground truth for all kernel correctness tests.

GQA: H_kv must divide H, and query head h reads K/V head h // (H // H_kv).
H_kv == H is standard multi-head attention, H_kv == 1 is multi-query attention.
"""

import math
import torch
import torch.nn.functional as F


def _expand_kv(k32, v32, num_q_heads):
    """GQA: repeat each K/V head so query head h uses K/V head h // (H // H_kv)."""
    H_kv = k32.shape[1]
    if v32.shape[1] != H_kv:
        raise ValueError(f"K/V head count mismatch: K={H_kv}, V={v32.shape[1]}")
    if H_kv < 1 or num_q_heads % H_kv != 0:
        raise ValueError(f"H={num_q_heads} not divisible by H_kv={H_kv}")
    group = num_q_heads // H_kv
    if group == 1:
        return k32, v32  # MHA: no copy
    return (k32.repeat_interleave(group, dim=1),
            v32.repeat_interleave(group, dim=1))


def reference_gqa_attention(q, k, v):
    q32, k32, v32 = q.float(), k.float(), v.float()
    k32, v32 = _expand_kv(k32, v32, q32.shape[1])
    return F.scaled_dot_product_attention(q32, k32, v32, is_causal=True)


def reference_gqa_attention_explicit(q, k, v):
    q32, k32, v32 = q.float(), k.float(), v.float()
    k32, v32 = _expand_kv(k32, v32, q32.shape[1])
    N, D = q32.shape[-2], q32.shape[-1]
    S = (q32 @ k32.transpose(-1, -2)) / math.sqrt(D)
    future = torch.triu(torch.ones(N, N, dtype=torch.bool, device=q.device), diagonal=1)
    S = S.masked_fill(future, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return P @ v32


# name imported by run_attention_gqa.py
causal_attention_reference_gqa = reference_gqa_attention_explicit
