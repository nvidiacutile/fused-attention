"""Correctness runner: our kernel vs the FP32 PyTorch reference (Task 5, owner: Harshith).

    python tests/run_correctness.py --impl unfused
    python tests/run_correctness.py --impl fused --block-m 64 --block-n 64
    python tests/run_correctness.py --impl fused --filter N512      # subset by case id

For each case in tests/cases.py (fixed seeds):
  1. build Q, K, V and validate them (attention/validation.py)
  2. run the chosen kernel and the PyTorch reference (FP32 intermediates)
  3. FAIL if the kernel output has NaN/Inf (softmax overflow or masking bugs)
  4. PASS if every element satisfies  |out - ref| <= ATOL + RTOL * |ref|
Prints max absolute error, max relative error and PASS/FAIL per case, writes a CSV,
and ends with an overall verdict.

Exit status: 0 all cases pass, 1 any case fails, 2 setup error.
"""

import argparse
import csv
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from attention.validation import validate_qkv  # noqa: E402
from run_attention import get_backend  # noqa: E402
from tests.cases import all_cases, make_inputs  # noqa: E402

ATOL = 2e-2
RTOL = 2e-2
REL_FLOOR = 1e-3  # relative error uses max(|ref|, REL_FLOOR) to avoid dividing by ~0
REFERENCE_BUDGET_BYTES = 2 * 1024**3  # cap on the reference's N x N FP32 scores per chunk

IMPL_TO_BACKEND = {"fused": "cutile", "unfused": "unfused", "reference": "reference"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Check a kernel against the FP32 PyTorch reference.")
    p.add_argument("--impl", required=True, choices=IMPL_TO_BACKEND,
                   help="kernel to test ('reference' checks the harness against itself)")
    p.add_argument("--block-m", type=int, default=64, help="query tile size (default 64)")
    p.add_argument("--block-n", type=int, default=64, help="key tile size (default 64)")
    p.add_argument("--filter", default="", help="only run cases whose id contains this text")
    p.add_argument("--csv", default=None, help="results CSV (default results/correctness_<impl>.csv)")
    p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)  # cpu only for harness self-tests
    return p.parse_args(argv)


def reference_chunked(reference_fn, q, k, v):
    """Run the reference over head chunks so its N x N scores fit in REFERENCE_BUDGET_BYTES."""
    B, H, N, _ = q.shape
    per_head = N * N * 4 * 2  # FP32 scores + probabilities
    chunk = max(1, min(H, REFERENCE_BUDGET_BYTES // per_head))
    out = torch.empty(q.shape[:3] + (v.shape[3],), dtype=torch.float32, device=q.device)
    for b in range(B):
        for h0 in range(0, H, chunk):
            sl = (slice(b, b + 1), slice(h0, h0 + chunk))
            out[sl] = reference_fn(q[sl], k[sl], v[sl]).float()
    return out


def compare(out, ref):
    """Return (max_abs, max_rel, worst tolerance ratio, nonfinite count). Pass iff ratio <= 1."""
    out = out.float()
    nonfinite = int((~torch.isfinite(out)).sum())
    err = (out - ref).abs()
    if nonfinite:
        return float("inf"), float("inf"), float("inf"), nonfinite
    max_abs = float(err.max())
    max_rel = float((err / ref.abs().clamp_min(REL_FLOOR)).max())
    ratio = float((err / (ATOL + RTOL * ref.abs())).max())
    return max_abs, max_rel, ratio, 0


def run_case(case, kernel_fn, reference_fn, args):
    q_np, k_np, v_np = make_inputs(case)
    validate_qkv(q_np, k_np, v_np, args.block_m, args.block_n)
    q, k, v = (torch.from_numpy(x).to(args.device) for x in (q_np, k_np, v_np))

    ref = reference_chunked(reference_fn, q, k, v)
    if not torch.isfinite(ref).all():
        raise RuntimeError("reference output has NaN/Inf (harness or reference bug)")

    t0 = time.perf_counter()
    out = kernel_fn(q, k, v, args.block_m, args.block_n)
    if args.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    expected = tuple(q.shape[:3]) + (v.shape[3],)
    if tuple(out.shape) != expected:
        raise RuntimeError(f"output shape {tuple(out.shape)}, expected {expected}")
    return compare(out, ref) + (seconds,)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("error: no CUDA GPU visible to PyTorch", file=sys.stderr)
        return 2
    try:
        kernel_fn = get_backend(IMPL_TO_BACKEND[args.impl])
        reference_backend = get_backend("reference")
        reference_fn = lambda q, k, v: reference_backend(q, k, v, None, None)  # noqa: E731
    except (ImportError, NotImplementedError) as e:
        print(f"error: could not load implementation: {e}", file=sys.stderr)
        return 2

    cases = [c for c in all_cases() if args.filter in c.id]
    if not cases:
        print(f"error: no cases match --filter {args.filter!r}", file=sys.stderr)
        return 2

    csv_path = args.csv or os.path.join(REPO_ROOT, "results", f"correctness_{args.impl}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    gpu = torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
    print(f"impl={args.impl}  tiles={args.block_m}x{args.block_n}  device={gpu}  "
          f"tolerance: |out-ref| <= {ATOL} + {RTOL}*|ref|  cases={len(cases)}\n")
    header = f"{'case':28s} {'max_abs_err':>11s} {'max_rel_err':>11s} {'tol_ratio':>9s} {'time_s':>7s}  result"
    print(header)
    print("-" * len(header))

    rows, failed = [], 0
    for case in cases:
        try:
            max_abs, max_rel, ratio, nonfinite, seconds = run_case(case, kernel_fn, reference_fn, args)
            passed = nonfinite == 0 and ratio <= 1.0
            note = f"{nonfinite} NaN/Inf values" if nonfinite else ""
        except NotImplementedError as e:
            print(f"\nerror: not implemented yet: {str(e) or 'a stub raised NotImplementedError'}. "
                  f"Implement attention/reference.py and the {args.impl} kernel first.", file=sys.stderr)
            return 2
        except Exception as e:  # report and continue: one broken case must not hide the rest
            max_abs = max_rel = ratio = seconds = float("nan")
            passed, note = False, f"{type(e).__name__}: {e}"
        failed += not passed
        result = "PASS" if passed else "FAIL"
        print(f"{case.id:28s} {max_abs:11.3e} {max_rel:11.3e} {ratio:9.3f} {seconds:7.3f}  {result}"
              + (f"  ({note})" if note else ""), flush=True)
        rows.append({"case": case.id, "B": case.B, "H": case.H, "N": case.N, "D": case.D,
                     "magnitude": case.magnitude, "seed": case.seed, "max_abs_err": max_abs,
                     "max_rel_err": max_rel, "tol_ratio": ratio, "time_s": seconds,
                     "result": result, "note": note})
        if args.device == "cuda":
            torch.cuda.empty_cache()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nresults written to {os.path.relpath(csv_path, REPO_ROOT)}")
    if failed:
        print(f"INCORRECT: {failed} of {len(cases)} cases failed for impl={args.impl}")
        return 1
    print(f"CORRECT: all {len(cases)} cases passed for impl={args.impl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
