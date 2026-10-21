"""cuTile smoke test: vector add on the GPU. Exits nonzero on failure.

Run by setup.sh after installing requirements to confirm that torch, the
driver and cuda-tile all work together on this instance.
"""

import sys

import torch
import cuda.tile as ct

TILE = 128


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: torch cannot see a CUDA GPU", file=sys.stderr)
        return 1

    n = 1 << 20
    torch.manual_seed(0)
    a = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)

    ct.launch(torch.cuda.current_stream(), (n // TILE, 1, 1), vector_add, (a, b, c, TILE))
    torch.cuda.synchronize()

    err = (c - (a + b)).abs().max().item()
    if torch.allclose(c, a + b):
        print(f"OK: cuTile kernel ran on {torch.cuda.get_device_name()} (max error {err})")
        return 0
    print(f"FAIL: cuTile result mismatch (max error {err})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
