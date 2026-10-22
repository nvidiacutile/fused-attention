"""Correctness test cases.

Defines the fixed list of 60 cases and builds reproducible Q, K, V for each.

Grid
  D in {32, 64, 128} x N in {512, 4096, 8192} x {tiny, large} x (B, H) in {(1,1), (2,8), (3,5)}  -> 54
  max shape (B, H) = (16, 32) at N = 512:  D in {32, 64, 128} x {tiny, large}               ->  6

Score magnitude
  Q and K are standard normal scaled by sigma, so scores q.k/sqrt(D) have std ~ sigma^2
  for every D. V is left unscaled so outputs stay O(1) and one tolerance fits all cases.
    tiny  sigma = 0.03 -> score std ~ 9e-4  (near-uniform softmax)
    large sigma = 7.0  -> score std ~ 49, max |score| ~ 240; exp(240) overflows FP32, so
                          only a max-subtracted (online) softmax gives finite results

Reproducibility
  Inputs are generated on the CPU with NumPy from a per-case seed derived from the case id
  (crc32, not Python's hash(), which changes between runs). Every machine and run gets
  identical inputs, and a single failing case can be regenerated on its own.
"""

import zlib
from dataclasses import dataclass

import numpy as np

HEAD_DIMS = (32, 64, 128)
SEQ_LENS = (512, 4096, 8192)
BATCH_HEADS = ((1, 1), (2, 8), (3, 5))
MAX_BATCH_HEADS = (16, 32)
MAX_SHAPE_SEQ_LEN = 512
MAGNITUDES = {"tiny": 0.03, "large": 7.0}


@dataclass(frozen=True)
class Case:
    B: int
    H: int
    N: int
    D: int
    magnitude: str  # "tiny" or "large"

    @property
    def id(self) -> str:
        return f"B{self.B}_H{self.H}_N{self.N}_D{self.D}_{self.magnitude}"

    @property
    def sigma(self) -> float:
        return MAGNITUDES[self.magnitude]

    @property
    def seed(self) -> int:
        return zlib.crc32(self.id.encode())


def all_cases() -> list:
    """The full, fixed, ordered list of correctness cases."""
    cases = [
        Case(B, H, N, D, mag)
        for D in HEAD_DIMS
        for N in SEQ_LENS
        for mag in MAGNITUDES
        for B, H in BATCH_HEADS
    ]
    B, H = MAX_BATCH_HEADS
    cases += [Case(B, H, MAX_SHAPE_SEQ_LEN, D, mag) for D in HEAD_DIMS for mag in MAGNITUDES]
    return cases


def make_inputs(case: Case):
    """Return contiguous float16 NumPy arrays Q, K, V of shape [B, H, N, D] for this case."""
    rng = np.random.default_rng(case.seed)
    shape = (case.B, case.H, case.N, case.D)

    def normal():
        return rng.standard_normal(shape, dtype=np.float32)

    q = (normal() * case.sigma).astype(np.float16)
    k = (normal() * case.sigma).astype(np.float16)
    v = normal().astype(np.float16)
    return q, k, v


if __name__ == "__main__":
    cases = all_cases()
    print(f"{len(cases)} cases")
    for c in cases:
        print(f"  {c.id:28s} seed={c.seed}")
