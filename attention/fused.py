import math
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def fused_attn_kernel(Q, K, V, O, scale: float, H: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    i = ct.bid(0)                 # which Q tile this block owns
    bh = ct.bid(1)                # which (batch, head)
    b = bh // H
    h = bh % H

    # load my Q tile ONCE
    q = ct.load(Q, index=(b, h, i, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    # running values, one per query row (FP32) (for each Q tile)
    m = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    rows = i * TILE_M + ct.arange(TILE_M, dtype=np.int32)[:, None]  # which query rows does this block/tile handle 

    for j in range(i + 1):        # causal: only K/V tiles 0..i (assumes TILE_M == TILE_N)
        kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))

        # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
        s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        s = ct.mma(q, kt,s) *scale
        

        # 2. causal mask on the diagonal tile only
        if j == i:
            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
            s = ct.where(cols > rows, -np.inf, s)

        # 3. new running max: bigger of old m and this tile's row max
        m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True))

        # correction factor for the old totals: e^(m - m_new)
        alpha = ct.exp(m - m_new)

        # this tile's numerators: e^(s - m_new)
        p =  ct.exp(s - m_new)

        # 6. update the running sum and the accumulator
        l = l*alpha  + ct.sum(p, axis=-1, keepdims=True)
        acc = ct.mma(p.astype(np.float16), v, acc * alpha)

        m = m_new

    out = acc / l
    ct.store(O, index=(b, h, i, 0), tile=out.astype(O.dtype).reshape((1, 1, TILE_M, TILE_D)))


def fused_attention(q, k, v, tile_m=64, tile_n=64):
    if tile_m != tile_n:
        raise NotImplementedError("fused kernel currently requires block_m == block_n")
    B, H, N, D = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    O = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    ct.launch(torch.cuda.current_stream(), (N // tile_m, B * H, 1), fused_attn_kernel,
              (q, k, v, O, scale, H, tile_m, tile_n, D))
    return O
