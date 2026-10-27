"""Unfused cuTile causal attention baseline.

FP16 inputs, FP32 accumulation. Three separate kernels:
  1. masked, scaled QKᵀ  -> scores written to global memory
  2. row-wise softmax    -> probabilities written to global memory
  3. P @ V               -> output
"""

import math
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def scores_kernel(Q, K, S, scale: float,
                  H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    bh = ct.bid(2)
    b = bh // H
    h = bh % H

    q = ct.load(Q, index=(b, h, bid_m, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
    kt = ct.load(K, index=(b, h, 0, bid_n), shape=(1, 1, TILE_D, TILE_N),
                 order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))

    s = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
    s = ct.mma(q, kt, s)
    s = s * scale

    rows = bid_m * TILE_M + ct.arange(TILE_M, dtype=np.int32)[:, None]
    cols = bid_n * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
    s = ct.where(rows >= cols, s, -np.inf)

    ct.store(S, index=(b, h, bid_m, bid_n), tile=s.reshape((1, 1, TILE_M, TILE_N)))


@ct.kernel
def softmax_kernel(S, P, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    bh = ct.bid(1)
    b = bh // H
    h = bh % H
    num_n = ct.cdiv(S.shape[3], TILE_N)

    m = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    for j in range(num_n):
        s = ct.load(S, index=(b, h, bid_m, j), shape=(1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        m = ct.maximum(m, ct.max(s, axis=-1, keepdims=True))

    l = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    for j in range(num_n):
        s = ct.load(S, index=(b, h, bid_m, j), shape=(1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        l = l + ct.sum(ct.exp(s - m), axis=-1, keepdims=True)

    for j in range(num_n):
        s = ct.load(S, index=(b, h, bid_m, j), shape=(1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        p = ct.exp(s - m) / l
        p = p.astype(P.dtype).reshape((1, 1, TILE_M, TILE_N))
        ct.store(P, index=(b, h, bid_m, j), tile=p)


@ct.kernel
def pv_kernel(P, V, O, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bid_m = ct.bid(0)
    bh = ct.bid(1)
    b = bh // H
    h = bh % H
    num_n = ct.cdiv(P.shape[3], TILE_N)

    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)
    for j in range(num_n):
        p = ct.load(P, index=(b, h, bid_m, j), shape=(1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        acc = ct.mma(p, v, acc)
    ct.store(O, index=(b, h, bid_m, 0), tile=acc.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))


def unfused_attention(q, k, v, tile_m=64, tile_n=64, return_intermediates=False):
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.is_cuda and q.ndim == 4 and q.shape == k.shape and v.shape[:3] == q.shape[:3]
    B, H, N, D = q.shape
    if N % tile_m or N % tile_n:
        raise ValueError(f"N={N} must be divisible by tile sizes {tile_m} and {tile_n}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    S = torch.empty((B, H, N, N), device=q.device, dtype=torch.float32)
    P = torch.empty((B, H, N, N), device=q.device, dtype=torch.float16)
    O = torch.empty((B, H, N, v.shape[-1]), device=q.device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)
    stream = torch.cuda.current_stream()

    ct.launch(stream, (N // tile_m, N // tile_n, B * H), scores_kernel,
              (q, k, S, scale, H, tile_m, tile_n, D))
    ct.launch(stream, (N // tile_m, B * H, 1), softmax_kernel,
              (S, P, H, tile_m, tile_n))
    ct.launch(stream, (N // tile_m, B * H, 1), pv_kernel,
              (P, v, O, H, tile_m, tile_n, v.shape[-1]))

    if return_intermediates:
        return O, S, P
    return O
