"""Fused causal attention with a recompute-based backward pass (FlashAttention-2 style) and autograd.

Forward  = v3 kernel (heavy-first order, exp2 + folded scale, FTZ/APPROX, occupancy=2) that also
           stores the per-row softmax statistics m and l, so the backward can rebuild P.
Backward = two cuTile kernels, no atomics and no in-register transposes:
  dkdv kernel: one block per (K/V tile j, b*h); loops over the query tiles that can see tile j and
               accumulates dV = P^T dO and dK = scale * dS^T Q, computing S^T = K Q^T directly.
  dq kernel:   one block per (Q tile i, b*h); loops over the visible K/V tiles (same order as the
               forward) and accumulates dQ = scale * dS K.
  with dS = P * (dP - Delta), dP = dO V^T, Delta = rowsum(dO * O) (computed in PyTorch).

Shapes: q, k, v are (B, H, N, D) float16, contiguous; N divisible by both tile sizes.
Use:    o = fused_attention_autograd(q, k, v, tile_m=64, tile_n=64); o.backward(do)
"""
import math

import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------- forward
@ct.kernel(occupancy=2)
def fwd_kernel(Q, K, V, O, M, L, scale: float,
               H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bh = ct.bid(0)
    i = ct.num_blocks(1) - 1 - ct.bid(1)          # heavy-first: last query tile first
    b = bh // H
    h = bh % H

    q = ct.load(Q, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
    m = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)
    rows = i * TILE_M + ct.arange(TILE_M, dtype=np.int32)[:, None]

    num_kv_tiles = ((i + 1) * TILE_M + TILE_N - 1) // TILE_N
    mask_start = (i * TILE_M) // TILE_N
    for j in range(num_kv_tiles):
        kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        s = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        s = ct.mma(q, kt, s)
        if j >= mask_start:
            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
            s = ct.where(cols > rows, -np.inf, s)
        m_new = ct.maximum(m, ct.max(s, axis=-1, keepdims=True) * scale)
        alpha = ct.exp2(m - m_new, flush_to_zero=True)
        p = ct.exp2(s * scale - m_new, flush_to_zero=True)
        l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
        acc = ct.mma(p.astype(np.float16), v, acc * alpha)
        m = m_new

    out = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)
    ct.store(O, index=(b, h, i, 0), tile=out.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))
    # softmax statistics for the backward (m is in scaled log2 units)
    ct.store(M, index=(b, h, i), tile=m.reshape((1, 1, TILE_M)))
    ct.store(L, index=(b, h, i), tile=l.reshape((1, 1, TILE_M)))


# ----------------------------------------------------------------------------- backward: dK, dV
@ct.kernel
def bwd_dkdv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, scale: float, sm_scale: float, n_q_tiles: int,
                    H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bh = ct.bid(0)
    j = ct.bid(1)                                  # K/V tile; j = 0 sees every query: heaviest first
    b = bh // H
    h = bh % H

    k = ct.load(K, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
    v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
    dk = ct.full((TILE_N, TILE_D), 0.0, dtype=np.float32)
    dv = ct.full((TILE_N, TILE_D), 0.0, dtype=np.float32)
    keys = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[:, None]          # (TILE_N, 1)

    i_start = (j * TILE_N) // TILE_M               # first query tile containing a query >= key tile start
    for t in range(n_q_tiles - i_start):
        i = i_start + t
        qt = ct.load(Q, index=(b, h, 0, i), shape=(1, 1, TILE_D, TILE_M),
                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_M))
        q = ct.load(Q, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
        do = ct.load(DO, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
        dot = ct.load(DO, index=(b, h, 0, i), shape=(1, 1, TILE_D, TILE_M),
                      order=(0, 1, 3, 2)).reshape((TILE_D, TILE_M))
        lse = ct.load(LSE, index=(b, h, i), shape=(1, 1, TILE_M)).reshape((1, TILE_M))
        delta = ct.load(DELTA, index=(b, h, i), shape=(1, 1, TILE_M)).reshape((1, TILE_M))

        st = ct.full((TILE_N, TILE_M), 0.0, dtype=np.float32)
        st = ct.mma(k, qt, st)                     # S^T = K Q^T  (keys x queries)
        if i * TILE_M < (j + 1) * TILE_N:          # tile touches the diagonal
            qidx = i * TILE_M + ct.arange(TILE_M, dtype=np.int32)[None, :]
            st = ct.where(keys > qidx, -np.inf, st)
        pt = ct.exp2(st * scale - lse, flush_to_zero=True)                  # P^T

        dv = ct.mma(pt.astype(np.float16), do, dv)                          # dV += P^T dO
        dpt = ct.full((TILE_N, TILE_M), 0.0, dtype=np.float32)
        dpt = ct.mma(v, dot, dpt)                                           # dP^T = V dO^T
        dst = pt * (dpt - delta)                                            # dS^T
        dk = ct.mma(dst.astype(np.float16), q, dk)                          # dK += dS^T Q

    dk = dk * sm_scale
    ct.store(DK, index=(b, h, j, 0), tile=dk.astype(DK.dtype).reshape((1, 1, TILE_N, TILE_D)))
    ct.store(DV, index=(b, h, j, 0), tile=dv.astype(DV.dtype).reshape((1, 1, TILE_N, TILE_D)))


# ----------------------------------------------------------------------------- backward: dQ
@ct.kernel
def bwd_dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, scale: float, sm_scale: float,
                  H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bh = ct.bid(0)
    i = ct.num_blocks(1) - 1 - ct.bid(1)          # heavy-first
    b = bh // H
    h = bh % H

    q = ct.load(Q, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
    do = ct.load(DO, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
    lse = ct.load(LSE, index=(b, h, i), shape=(1, 1, TILE_M)).reshape((TILE_M, 1))
    delta = ct.load(DELTA, index=(b, h, i), shape=(1, 1, TILE_M)).reshape((TILE_M, 1))
    dq = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)
    rows = i * TILE_M + ct.arange(TILE_M, dtype=np.int32)[:, None]

    num_kv_tiles = ((i + 1) * TILE_M + TILE_N - 1) // TILE_N
    mask_start = (i * TILE_M) // TILE_N
    for j in range(num_kv_tiles):
        kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
        k = ct.load(K, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        vt = ct.load(V, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))

        s = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        s = ct.mma(q, kt, s)
        if j >= mask_start:
            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
            s = ct.where(cols > rows, -np.inf, s)
        p = ct.exp2(s * scale - lse, flush_to_zero=True)                    # P
        dp = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        dp = ct.mma(do, vt, dp)                                             # dP = dO V^T
        ds = p * (dp - delta)                                               # dS
        dq = ct.mma(ds.astype(np.float16), k, dq)                           # dQ += dS K

    dq = dq * sm_scale
    ct.store(DQ, index=(b, h, i, 0), tile=dq.astype(DQ.dtype).reshape((1, 1, TILE_M, TILE_D)))


# ----------------------------------------------------------------------------- host side
def _check(q, k, v, tile_m, tile_n):
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError("q, k, v must be float16")
    if q.shape != k.shape or q.shape != v.shape or q.dim() != 4:
        raise ValueError(f"q, k, v must all be (B, H, N, D); got {q.shape}, {k.shape}, {v.shape}")
    N = q.shape[2]
    if N % tile_m or N % tile_n:
        raise ValueError(f"N={N} must be divisible by tile_m={tile_m} and tile_n={tile_n}")


def _forward(q, k, v, tile_m, tile_n):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    m = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    l = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    scale = LOG2E / math.sqrt(D)
    ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fwd_kernel,
              (q, k, v, o, m, l, scale, H, tile_m, tile_n, D))
    lse = m + torch.log2(l)                        # log2-domain log-sum-exp of the scaled scores
    return o, lse


def _backward(q, k, v, o, lse, do, tile_m, tile_n):
    B, H, N, D = q.shape
    delta = (do.float() * o.float()).sum(-1).contiguous()     # (B, H, N) fp32
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    scale = LOG2E / math.sqrt(D)
    sm_scale = 1.0 / math.sqrt(D)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (B * H, N // tile_n, 1), bwd_dkdv_kernel,
              (q, k, v, do, lse, delta, dk, dv, scale, sm_scale, N // tile_m, H, tile_m, tile_n, D))
    ct.launch(stream, (B * H, N // tile_m, 1), bwd_dq_kernel,
              (q, k, v, do, lse, delta, dq, scale, sm_scale, H, tile_m, tile_n, D))
    return dq, dk, dv


class FusedAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, tile_m=64, tile_n=64):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        _check(q, k, v, tile_m, tile_n)
        o, lse = _forward(q, k, v, tile_m, tile_n)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.tiles = (tile_m, tile_n)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _backward(q, k, v, o, lse, do.contiguous(), *ctx.tiles)
        return dq, dk, dv, None, None


def fused_attention_autograd(q, k, v, tile_m=64, tile_n=64):
    """Causal attention (B, H, N, D) float16 with autograd support."""
    return FusedAttentionFn.apply(q, k, v, tile_m, tile_n)
