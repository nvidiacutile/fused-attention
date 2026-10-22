"""Command-line interface for causal attention (Task 4, owner: Harshith).

    python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile

Input:  .npz with contiguous float16 arrays
          Q [B, H, N, D], K [B, H, N, D], V [B, H, N, Dv]
        limits: 1<=B<=16, 1<=H<=32, 1<=N<=8192, D, Dv in {32, 64, 128},
        N divisible by --block-m and --block-n
Output: .npy, float16, shape [B, H, N, Dv]

Backends:
  cutile     fused cuTile kernel (attention/fused.py)
  unfused    unfused cuTile baseline (attention/unfused.py)
  reference  PyTorch reference, FP32 intermediates (attention/reference.py)
  cutedsl    optional CuTe DSL track (not implemented)

Exit status: 0 success, 2 unsupported input or usage error, 1 runtime failure.
"""

import argparse
import sys
import time

import numpy as np

from attention.validation import ValidationError, validate_qkv

OUTPUT_DTYPE = np.float16
EXIT_OK, EXIT_RUNTIME, EXIT_INVALID = 0, 1, 2

BACKENDS = ("cutile", "unfused", "reference", "cutedsl")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Causal scaled dot-product attention on Q, K, V from an .npz file.")
    p.add_argument("--input", required=True, help="input .npz containing Q, K, V")
    p.add_argument("--output", required=True, help="output .npy path")
    p.add_argument("--backend", choices=BACKENDS, default="cutile", help="implementation to run (default: cutile)")
    p.add_argument("--block-m", type=int, default=64, help="query tile size (default: 64)")
    p.add_argument("--block-n", type=int, default=64, help="key/value tile size (default: 64)")
    p.add_argument("--verbose", action="store_true", help="print shapes and timing to stderr")
    return p.parse_args(argv)


def load_qkv(path):
    """Load Q, K, V from an .npz file, raising ValidationError for missing or malformed files."""
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in ("Q", "K", "V") if key not in data.files]
            if missing:
                raise ValidationError(
                    f"{path} is missing array(s) {', '.join(missing)}; found {sorted(data.files)}"
                )
            return data["Q"], data["K"], data["V"]
    except FileNotFoundError:
        raise ValidationError(f"input file not found: {path}")
    except (OSError, ValueError) as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"could not read {path} as .npz: {e}")


def get_backend(name):
    """Return fn(q, k, v, block_m, block_n) -> torch.Tensor [B, H, N, Dv]. Imports lazily."""
    if name == "cutile":
        from attention.fused import fused_attention
        return fused_attention
    if name == "unfused":
        from attention.unfused import unfused_attention
        return lambda q, k, v, bm, bn: unfused_attention(q, k, v, tile_m=bm, tile_n=bn)
    if name == "reference":
        from attention.reference import causal_attention_reference
        return lambda q, k, v, bm, bn: causal_attention_reference(q, k, v)
    raise NotImplementedError(f"backend '{name}' is not implemented")


def run(q, k, v, backend, block_m, block_n):
    fn = get_backend(backend)  # resolve first so unimplemented backends fail fast
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU visible to PyTorch")
    q_t, k_t, v_t = (torch.from_numpy(x).cuda() for x in (q, k, v))
    out = fn(q_t, k_t, v_t, block_m, block_n)
    torch.cuda.synchronize()
    return out.to(torch.float16).cpu().numpy()


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        q, k, v = load_qkv(args.input)
        validate_qkv(q, k, v, args.block_m, args.block_n)
    except ValidationError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INVALID

    if args.verbose:
        print(f"Q{q.shape} K{k.shape} V{v.shape} backend={args.backend} "
              f"block_m={args.block_m} block_n={args.block_n}", file=sys.stderr)

    try:
        t0 = time.perf_counter()
        out = run(q, k, v, args.backend, args.block_m, args.block_n)
        elapsed = time.perf_counter() - t0
    except NotImplementedError as e:
        print(f"error: {str(e) or f'backend {args.backend!r} is not implemented yet'}", file=sys.stderr)
        return EXIT_RUNTIME
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("attention."):
            print(f"error: backend '{args.backend}' is not implemented yet ({e.name} missing)", file=sys.stderr)
        else:
            print(f"error: missing dependency '{e.name}'; run setup.sh and activate .venv", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as e:  # kernel/compile/driver failures
        print(f"error: {args.backend} backend failed: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_RUNTIME

    expected = q.shape[:3] + (v.shape[3],)
    if out.shape != expected:
        print(f"error: backend returned shape {out.shape}, expected {expected}", file=sys.stderr)
        return EXIT_RUNTIME
    if not np.isfinite(out).all():
        print("error: output contains NaN or Inf", file=sys.stderr)
        return EXIT_RUNTIME

    np.save(args.output, out.astype(OUTPUT_DTYPE, copy=False))
    if args.verbose:
        print(f"wrote {args.output} {out.shape} {OUTPUT_DTYPE.__name__} in {elapsed:.3f}s "
              f"(includes transfer and any JIT compile)", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
