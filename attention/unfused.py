"""Unfused cuTile causal attention baseline.

FP16 inputs, FP32 accumulation. One block per (query row, batch*head).
1. masked, scaled q·Kᵀ  -> scores row written to S
2. row softmax          -> probabilities row written to P
3. p @ V                -> output row
"""

import math
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def scores_kernel(Q, K, S, scale: float, H: ConstInt, N: ConstInt, D: ConstInt):
    i = ct.bid(0)                      # query row
    bh = ct.bid(1)
    b, h = bh // H, bh % H

    q = ct.load(Q, index=(b, h, i, 0), shape=(1, 1, 1, D)).reshape((D, 1)).astype(np.float32)
    kt = ct.load(K, index=(b, h, 0, 0), shape=(1, 1, D, N),
                 order=(0, 1, 3, 2)).reshape((D, N)).astype(np.float32)

    s = ct.sum(q * kt, axis=0, keepdims=True) * scale        # (1, N)

    cols = ct.arange(N, dtype=np.int32)[None, :]
    s = ct.where(cols <= i, s, -np.inf)                      # causal mask

    ct.store(S, index=(b, h, i, 0), tile=s.reshape((1, 1, 1, N)))


@ct.kernel
def softmax_kernel(S, P, H: ConstInt, N: ConstInt):
    i = ct.bid(0)
    bh = ct.bid(1)
    b, h = bh // H, bh % H

    s = ct.load(S, index=(b, h, i, 0), shape=(1, 1, 1, N)).reshape((1, N))
    m = ct.max(s, axis=-1, keepdims=True)
    e = ct.exp(s - m)
    p = e / ct.sum(e, axis=-1, keepdims=True)

    ct.store(P, index=(b, h, i, 0), tile=p.astype(P.dtype).reshape((1, 1, 1, N)))


@ct.kernel
def pv_kernel(P, V, O, H: ConstInt, N: ConstInt, D: ConstInt):
    i = ct.bid(0)
    bh = ct.bid(1)
    b, h = bh // H, bh % H

    p = ct.load(P, index=(b, h, i, 0), shape=(1, 1, 1, N)).reshape((N, 1)).astype(np.float32)
    v = ct.load(V, index=(b, h, 0, 0), shape=(1, 1, N, D)).reshape((N, D)).astype(np.float32)

    o = ct.sum(p * v, axis=0, keepdims=True)                 # (1, D)
    ct.store(O, index=(b, h, i, 0), tile=o.astype(O.dtype).reshape((1, 1, 1, D)))


def unfused_attention(q, k, v, return_intermediates=False, **_ignored):
    B, H, N, D = q.shape
    Dv = v.shape[-1]
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    S = torch.empty((B, H, N, N), device=q.device, dtype=torch.float32)
    P = torch.empty((B, H, N, N), device=q.device, dtype=torch.float16)
    O = torch.empty((B, H, N, Dv), device=q.device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)
    stream = torch.cuda.current_stream()
    grid = (N, B * H, 1)

    ct.launch(stream, grid, scores_kernel, (q, k, S, scale, H, N, D))
    ct.launch(stream, grid, softmax_kernel, (S, P, H, N))
    ct.launch(stream, grid, pv_kernel, (P, v, O, H, N, Dv))

    if return_intermediates:
        return O, S, P
    return O
