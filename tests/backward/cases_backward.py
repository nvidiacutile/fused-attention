"""Cases for the fused-backward correctness test (tests/backward/run_correctness_backward.py).

Defines the fixed list of 7 backward cases and builds reproducible FP16 Q, K, V and dO
for each, plus the shapes used for the forward+backward timing table.

Cases
  (B, H, N, D, sigma): Q, K, V are standard normal scaled by sigma and dO is standard
  normal. Larger sigma gives larger attention scores; sigma = 7.0 is the stress case.

Reproducibility
  Inputs are drawn on the GPU from a torch.Generator seeded per case (seed = N + D),
  so every run gets identical inputs.
"""

from dataclasses import dataclass

import torch

# (B, H, N, D, sigma)
CASE_SHAPES = (
    (1, 1, 128, 64, 1.0),
    (2, 4, 256, 64, 1.0),
    (2, 4, 512, 128, 1.0),
    (1, 2, 1024, 64, 3.0),
    (2, 8, 2048, 64, 1.0),
    (2, 8, 2048, 128, 1.0),
    (1, 4, 512, 64, 7.0),
)

# forward+backward timing grid
BENCH_B, BENCH_H = 2, 8
BENCH_HEAD_DIMS = (64, 128)
BENCH_SEQ_LENS = (512, 2048, 8192)


@dataclass(frozen=True)
class Case:
    B: int
    H: int
    N: int
    D: int
    sigma: float

    @property
    def id(self) -> str:
        return f"B{self.B}_H{self.H}_N{self.N}_D{self.D}_s{self.sigma:g}"

    @property
    def seed(self) -> int:
        return self.N + self.D


def all_cases() -> list:
    """The full, fixed, ordered list of backward correctness cases."""
    return [Case(*shape) for shape in CASE_SHAPES]


def make(B, H, N, D, sigma, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(B, H, N, D, device="cuda", generator=g) * sigma for _ in range(3))
    do = torch.randn(B, H, N, D, device="cuda", generator=g)
    return [t.half() for t in (q, k, v)], do.half()


def make_inputs(case: Case):
    """Return ([q, k, v], dO) as FP16 CUDA tensors of shape [B, H, N, D] for this case."""
    return make(case.B, case.H, case.N, case.D, case.sigma, case.seed)


def bench_inputs(N, D):
    """FP16 inputs for one forward+backward timing point (sigma 1, seed 0)."""
    return make(BENCH_B, BENCH_H, N, D, 1.0, seed=0)


if __name__ == "__main__":
    cases = all_cases()
    print(f"{len(cases)} cases")
    for c in cases:
        print(f"  {c.id:28s} seed={c.seed}")
