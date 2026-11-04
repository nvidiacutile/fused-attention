# v9: v3 + lazy rescaling: rescale acc and l only when some row max grows by more than 8 (log2 units); otherwise keep the old max
import math
import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
LOG2E = 1.4426950408889634        # log2(e): exp(x) = exp2(x * LOG2E)
RESCALE_THRESHOLD = 8.0           # log2 units: p may grow up to 2^8 = 256 before we rescale


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

    for j in range(num_kv_tiles):
        kt = ct.load(K, index=(b, h, 0, j), shape=(1, 1, TILE_D, TILE_N),
                     order=(0, 1, 3, 2)).reshape((TILE_D, TILE_N))
        v = ct.load(V, index=(b, h, j, 0), shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))

        # 1. scores tile: q @ kt into an FP32 zero tile, then multiply by scale
        s = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        s = ct.mma(q, kt,s) 
        

        # 2. causal mask only on tiles that straddle/cross the diagonal
        if j >= mask_start:
            cols = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)[None, :]
            s = ct.where(cols > rows, -np.inf, s)

        # 3. candidate running max for this tile
        m_cand = ct.maximum(m, ct.max(s, axis=-1, keepdims=True) * scale)

        # 4. lazy rescaling: only when some row's max grew by more than the threshold.
        #    Otherwise keep the old (stale) max: p = 2^(s*scale - m) is then at most 2^8,
        #    and O = acc / l stays exact because acc and l use the same reference max.
        grow = ct.max(m_cand - m)                     # largest increase over the block's rows
        if grow > RESCALE_THRESHOLD:
            alpha = ct.exp2(m - m_cand, flush_to_zero=True)
            l = l * alpha
            acc = acc * alpha
            m = m_cand

        # 5. this tile's numerators relative to the (possibly stale) max
        p = ct.exp2(s * scale - m, flush_to_zero=True)

        # 6. update the running sum and the accumulator (no rescale multiply here)
        l = l + ct.sum(p, axis=-1, keepdims=True)
        acc = ct.mma(p.astype(np.float16), v, acc)

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
