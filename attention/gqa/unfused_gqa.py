"""Tiled unfused cuTile causal attention baseline, GQA/MQA version (MHA-only: attention/fused/unfused.py).

FP16 inputs, FP32 accumulation. One block per (query tile of TILE_M rows, batch*head).
Three separate kernels; S and P are materialized in global memory (that is what
makes it "unfused"), but every kernel works on TILE_M x TILE_N tiles with ct.mma:

1. scores_kernel:  S = causal_mask(Q @ K^T * scale)     FP32 tiles
2. softmax_kernel: P = row_softmax(S)                   pass 1: row max + row sum
                                                        pass 2: normalize, store FP16
3. pv_kernel:      O = P @ V                            FP32 accumulator

Only tiles on or below the causal diagonal are computed, read or written.
Tiles strictly above the diagonal of S / P are never touched; they are filled
with -inf / 0 only when return_intermediates=True so the returned S, P are clean.

Memory note: S (FP32) and P (FP16) are full B*H*N*N buffers.

GQA/MQA: Q is (B, H, N, D), K and V are (B, H_kv, N, D) with H % H_kv == 0.
Query head h reads K/V head h // QUERY_GROUP, where QUERY_GROUP = H // H_kv.
H_kv == H is MHA (QUERY_GROUP = 1), H_kv == 1 is MQA. S, P and O keep H heads.
"""

import math
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def scores_kernel(Q, K, S, scale: float, H: ConstInt, QUERY_GROUP: ConstInt,
                  TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    i = ct.bid(0)                      # query tile
    bh = ct.bid(1)
    b, h = bh // H, bh % H
    kv_h = h // QUERY_GROUP            # GQA: K/V head shared by this query head

    q = ct.load(Q, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
    rows = i * TILE_M + ct.arange(TILE_M, dtype=np.int32)[:, None]

    num_kv_tiles = ((i + 1) * TILE_M + TILE_N - 1) // TILE_N   # causal: skip future tiles
    mask_start = (i * TILE_M) // TILE_N                         # first tile touching the diagonal

    for j in range(num_kv_tiles):
        kt = ct.load(K, index=(b, kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
        s = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        s = ct.mma(q, kt, s) * scale
        if j >= mask_start:
            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
            s = ct.where(cols > rows, -np.inf, s)
        ct.store(S, index=(b, h, i, j), tile=s.reshape((1, 1, TILE_M, TILE_N)))


@ct.kernel
def softmax_kernel(S, P, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt):
    i = ct.bid(0)
    bh = ct.bid(1)
    b, h = bh // H, bh % H

    num_kv_tiles = ((i + 1) * TILE_M + TILE_N - 1) // TILE_N

    # pass 1: row max m and row sum l = sum(exp(s - m)), rescaled as m grows
    m = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    for j in range(num_kv_tiles):
        s = ct.load(S, index=(b, h, i, j), shape=(1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        m_new = ct.maximum(m, ct.max(s, axis=-1, keepdims=True))
        l = l * ct.exp(m - m_new) + ct.sum(ct.exp(s - m_new), axis=-1, keepdims=True)
        m = m_new

    # pass 2: normalize and write P
    for j in range(num_kv_tiles):
        s = ct.load(S, index=(b, h, i, j), shape=(1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        p = ct.exp(s - m) / l
        ct.store(P, index=(b, h, i, j), tile=p.astype(P.dtype).reshape((1, 1, TILE_M, TILE_N)))


@ct.kernel
def pv_kernel(P, V, O, H: ConstInt, QUERY_GROUP: ConstInt,
              TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    i = ct.bid(0)
    bh = ct.bid(1)
    b, h = bh // H, bh % H
    kv_h = h // QUERY_GROUP            # GQA: K/V head shared by this query head

    num_kv_tiles = ((i + 1) * TILE_M + TILE_N - 1) // TILE_N

    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)
    for j in range(num_kv_tiles):
        p = ct.load(P, index=(b, h, i, j), shape=(1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        v = ct.load(V, index=(b, kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        acc = ct.mma(p, v, acc)

    ct.store(O, index=(b, h, i, 0), tile=acc.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))


def unfused_gqa_attention(q, k, v, tile_m=64, tile_n=64, return_intermediates=False):
    B, H, N, D = q.shape
    if N % tile_m != 0 or N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_m={tile_m} and tile_n={tile_n}")
    H_kv = k.shape[1]
    if k.shape != (B, H_kv, N, D) or v.shape != k.shape or H % H_kv != 0:
        raise ValueError(f"GQA needs K, V of shape [B, H_kv, N, D] with H_kv dividing H; got "
                         f"Q {tuple(q.shape)}, K {tuple(k.shape)}, V {tuple(v.shape)}")
    query_group = H // H_kv                # query heads per K/V head
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    if return_intermediates:
        S = torch.full((B, H, N, N), float("-inf"), device=q.device, dtype=torch.float32)
        P = torch.zeros((B, H, N, N), device=q.device, dtype=torch.float16)
    else:
        S = torch.empty((B, H, N, N), device=q.device, dtype=torch.float32)
        P = torch.empty((B, H, N, N), device=q.device, dtype=torch.float16)
    O = torch.empty((B, H, N, D), device=q.device, dtype=q.dtype)

    scale = 1.0 / math.sqrt(D)
    stream = torch.cuda.current_stream()
    grid = (N // tile_m, B * H, 1)

    ct.launch(stream, grid, scores_kernel, (q, k, S, scale, H, query_group, tile_m, tile_n, D))
    ct.launch(stream, grid, softmax_kernel, (S, P, H, tile_m, tile_n))
    ct.launch(stream, grid, pv_kernel, (P, v, O, H, query_group, tile_m, tile_n, D))

    if return_intermediates:
        return O, S, P
    return O
