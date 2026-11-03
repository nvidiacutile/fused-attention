# v8: v3 + software-pipelined K/V loop: load K/V and compute S for tile j+1 before the softmax/PV of tile j (independent work in each iteration)
import math
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)


@ct.kernel(occupancy=2)
def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bh = ct.bid(0)                          # which (batch, head); axis 0 is launched fastest
    i = ct.num_blocks(1) - 1 - ct.bid(1)    # which Q tile; heaviest (last) Q tiles launch first
    b = bh // H                   # goes up by 1 every H [0 to BATCH-1]
    h = bh % H                    # varies from 0 to H-1 in every H

    # load my Q tile ONCE
    q = ct.load(Q, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    # running values, one per query row (FP32) (for each Q tile)
    m = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    rows = i * TILE_M + ct.arange(TILE_M, dtype=np.int32)[:, None]  # which query rows does this block/tile handle 

    # causal bounds in units of K/V tiles (works for TILE_M != TILE_N)
    # last query row of this tile is (i+1)*TILE_M - 1 -> need K/V tiles 0..ceil((i+1)*TILE_M / TILE_N)-1
    num_kv_tiles = ((i + 1) * TILE_M + TILE_N - 1) // TILE_N
    # first K/V tile that can contain a col > the first query row (i*TILE_M) -> needs masking
    mask_start = (i * TILE_M) // TILE_N

    # ---- prologue: scores for the first K/V tile
    kt = ct.load(K, index=(b, h, 0, 0), shape=(1, 1, TILE_D, TILE_N),
                 order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
    v_cur = ct.load(V, index=(b, h, 0, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
    s_cur = ct.mma(q, kt, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))

    # ---- steady state: tile j's softmax + P@V overlaps with tile j+1's loads and Q@K^T
    for j in range(num_kv_tiles - 1):
        # (a) next tile: independent of the current softmax, so it can run while (b) waits
        kt_n = ct.load(K, index=(b, h, 0, j + 1), shape=(1, 1, TILE_D, TILE_N),
                       order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
        v_n = ct.load(V, index=(b, h, j + 1, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        s_n = ct.mma(q, kt_n, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))

        # (b) current tile j: mask (diagonal only), online softmax, P@V
        if j >= mask_start:
            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
            s_cur = ct.where(cols > rows, -np.inf, s_cur)
        m_new = ct.maximum(m, ct.max(s_cur, axis=-1, keepdims=True) * scale)
        alpha = ct.exp2(m - m_new, flush_to_zero=True)
        p = ct.exp2(s_cur * scale - m_new, flush_to_zero=True)
        l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
        acc = ct.mma(p.astype(np.float16), v_cur, acc * alpha)
        m = m_new

        s_cur = s_n
        v_cur = v_n

    # ---- epilogue: last K/V tile (always on the diagonal, so always masked)
    j_last = num_kv_tiles - 1
    cols = j_last * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
    s_cur = ct.where(cols > rows, -np.inf, s_cur)
    m_new = ct.maximum(m, ct.max(s_cur, axis=-1, keepdims=True) * scale)
    alpha = ct.exp2(m - m_new, flush_to_zero=True)
    p = ct.exp2(s_cur * scale - m_new, flush_to_zero=True)
    l = l * alpha + ct.sum(p, axis=-1, keepdims=True)
    acc = ct.mma(p.astype(np.float16), v_cur, acc * alpha)

    out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
    ct.store(O, index=(b, h, i, 0), tile=out.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))


def fused_attention(q, k, v, tile_m=64, tile_n=64):
    B, H, N, D = q.shape
    if N % tile_m != 0 or N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_m={tile_m} and tile_n={tile_n}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    O = torch.empty_like(q)
    scale = LOG2E / math.sqrt(D)      # log2(e) / sqrt(D), folded into one constant
    ct.launch(torch.cuda.current_stream(), (B * H, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H, tile_m, tile_n, D))
    return O
