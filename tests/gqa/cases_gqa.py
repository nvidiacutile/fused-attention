"""GQA correctness test cases.

Defines the fixed list of 96 grouped-query attention cases and builds reproducible
Q, K, V for each. Q has H query heads; K and V have H_kv shared K/V heads, and each
group of H // H_kv query heads attends to one K/V head.

Grid
D in {32, 64, 128} x N in {128, 512, 2048, 8192} x {tiny, large} x (B, H) = (4, 8)
x H_kv in {1, 2, 4, 8} -> 96
H_kv = 1 is multi-query attention (MQA); H_kv = H = 8 is plain multi-head attention (MHA).

Score magnitude
Same as cases.py: Q and K are standard normal scaled by sigma, so scores q.k/sqrt(D)
have std ~ sigma^2 for every D. V is left unscaled so outputs stay O(1).
tiny sigma = 0.03 -> near-uniform softmax
large sigma = 7.0 -> max |score| ~ 240; only a max-subtracted (online) softmax stays finite

Reproducibility
Inputs are generated on the CPU with NumPy from a per-case seed derived from the case id
(crc32), so every machine and run gets identical inputs.
"""

import zlib
from dataclasses import dataclass

import numpy as np

HEAD_DIMS = (32, 64, 128)
SEQ_LENS = (128, 512, 2048, 8192)
BATCH = 4
NUM_Q_HEADS = 8
NUM_KV_HEADS = (1, 2, 4, 8)
MAGNITUDES = {"tiny": 0.03, "large": 7.0}


@dataclass(frozen=True)
class Case:
    B: int
    H: int
    H_kv: int
    N: int
    D: int
    magnitude: str  # "tiny" or "large"

    def __post_init__(self):
        if self.H % self.H_kv != 0:
            raise ValueError(f"H_kv={self.H_kv} must divide H={self.H}")

    @property
    def group(self) -> int:
        """Query heads per K/V head (H // H_kv)."""
        return self.H // self.H_kv

    @property
    def id(self) -> str:
        return f"B{self.B}_H{self.H}_Hkv{self.H_kv}_N{self.N}_D{self.D}_{self.magnitude}"

    @property
    def sigma(self) -> float:
        return MAGNITUDES[self.magnitude]

    @property
    def seed(self) -> int:
        return zlib.crc32(self.id.encode())


def all_cases() -> list:
    """The full, fixed, ordered list of GQA correctness cases."""
    return [
        Case(BATCH, NUM_Q_HEADS, H_kv, N, D, mag)
        for D in HEAD_DIMS
        for N in SEQ_LENS
        for mag in MAGNITUDES
        for H_kv in NUM_KV_HEADS
    ]


def make_inputs(case: Case):
    """Return contiguous float16 NumPy arrays Q [B, H, N, D], K and V [B, H_kv, N, D]."""
    rng = np.random.default_rng(case.seed)
    q_shape = (case.B, case.H, case.N, case.D)
    kv_shape = (case.B, case.H_kv, case.N, case.D)

    def normal(shape):
        return rng.standard_normal(shape, dtype=np.float32)

    q = (normal(q_shape) * case.sigma).astype(np.float16)
    k = (normal(kv_shape) * case.sigma).astype(np.float16)
    v = normal(kv_shape).astype(np.float16)
    return q, k, v


if __name__ == "__main__":
    cases = all_cases()
    print(f"{len(cases)} cases")
    for c in cases:
        print(f"  {c.id:36s} group={c.group}  seed={c.seed}")
