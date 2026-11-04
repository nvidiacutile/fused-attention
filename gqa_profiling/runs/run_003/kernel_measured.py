# v4: v3 + GQA head packing (one block handles all query heads sharing a K/V head)
# GQA: K/V have H_kv heads; query head h reads K/V head h // (H // H_kv).
import math
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)


@ct.kernel(occupancy=2)
def fused_attn_kernel(Q, K, V, O, scale: float, H_KV: ConstInt, QUERY_GROUP: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bkv = ct.bid(0)
    i = ct.num_blocks(1) - 1 - ct.bid(1)    # which Q tile; heaviest (last) Q tiles launch first
    b = bkv // H_KV
    kv_h = bkv % H_KV

    ROWS = QUERY_GROUP * TILE_M

    # load my Q tile ONCE
    q = ct.load(Q, index=(b, kv_h, i, 0), shape=(1, QUERY_GROUP, TILE_M, TILE_D)).reshape((ROWS, TILE_D))

    # running values, one per query row (FP32) (for each Q tile)
    m = ct.full((ROWS, 1), -np.inf, dtype=np.float32)
    l = ct.full((ROWS, 1), 0.0, dtype=np.float32)
    acc = ct.full((ROWS, TILE_D), 0.0, dtype=np.float32)

    pos = ct.full((QUERY_GROUP, TILE_M), 0, dtype=np.int32) + ct.arange(TILE_M, dtype=np.int32)[None, :]
    rows = i * TILE_M + pos.reshape((ROWS, 1))  # which query rows does this block/tile handle 

    # causal bounds in units of K/V tiles (works for TILE_M != TILE_N)
    # last query row of this tile is (i+1)*TILE_M - 1 -> need K/V tiles 0..ceil((i+1)*TILE_M / TILE_N)-1
    num_kv_tiles = ((i + 1) * TILE_M + TILE_N - 1) // TILE_N
    # first K/V tile that can contain a col > the first query row (i*TILE_M) -> needs masking
    mask_start = (i * TILE_M) // TILE_N

    for j in range(num_kv_tiles):
        kt = ct.load(K, index=(b, kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
        v = ct.load(V, index=(b, kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))

        # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
        s = ct.full((ROWS, TILE_N), 0.0, dtype=ct.float32)
        s = ct.mma(q, kt,s) 
        

        # 2. causal mask only on tiles that straddle/cross the diagonal
        if j >= mask_start:
            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
            s = ct.where(cols > rows, -np.inf, s)

        # 3. new running max: bigger of old m and this tile's row max
        m_new = ct.maximum(m, ct.max(s, axis = -1, keepdims = True)*scale)

        # correction factor for the old totals: e^(m - m_new)
        alpha = ct.exp2(m - m_new, flush_to_zero=True)

        # this tile's numerators: e^(s - m_new)
        p = ct.exp2(s * scale - m_new, flush_to_zero=True)   

        # 6. update the running sum and the accumulator
        l = l*alpha  + ct.sum(p, axis=-1, keepdims=True)
        acc = ct.mma(p.astype(np.float16), v, acc * alpha)

        m = m_new

    out   = ct.truediv(acc, l, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)   # instead of acc / l
    ct.store(O, index=(b, kv_h, i, 0), tile=out.astype(O.dtype).reshape((1, QUERY_GROUP, TILE_M, TILE_D)))


def fused_gqa_attention(q, k, v, tile_m=32, tile_n=64):
    B, H, N, D = q.shape
    H_kv = k.shape[1]
    if k.shape != (B, H_kv, N, D) or v.shape != k.shape or H % H_kv != 0:
        raise ValueError(f"GQA needs K, V of shape [B, H_kv, N, D] with H_kv dividing H; got Q {tuple(q.shape)}, K {tuple(k.shape)}, V {tuple(v.shape)}")
    query_group = H // H_kv                # query heads per K/V head
    if N % tile_m != 0 or N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_m={tile_m} and tile_n={tile_n}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    O = torch.empty_like(q)
    scale = LOG2E / math.sqrt(D)      # log2(e) / sqrt(D), folded into one constant
    ct.launch(torch.cuda.current_stream(), (B * H_kv, N // tile_m, 1), fused_attn_kernel, (q, k, v, O, scale, H_kv, query_group, tile_m, tile_n, D))
    return O
