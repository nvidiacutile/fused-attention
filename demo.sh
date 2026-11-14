#!/usr/bin/env bash
# Demo: (1) correctness of the fused kernel, (2) forward benchmark.
# Run from the repo root, with demo_forward.py next to it:   bash demo.sh
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null

KERNEL=attention/fused/fused_kernels/v3_occupancy2.py
TILE_M=64
TILE_N=64

banner() { echo; echo "==================== $1 ===================="; echo; }

# ------------------------------------------------------------------ 1. correctness
banner "1. Correctness: fused kernel vs FP32 PyTorch reference"

python - <<'EOF'
from tests.fused.cases import (BATCH_HEADS, MAX_BATCH_HEADS, SEQ_LENS, MAX_SHAPE_SEQ_LENS,
                               HEAD_DIMS, MAGNITUDES, all_cases)
print("Test grid we iterate over:")
print(f"  (B, H)  : {', '.join(str(bh) for bh in BATCH_HEADS)}")
print(f"  N       : {', '.join(map(str, SEQ_LENS))}")
print(f"  D       : {', '.join(map(str, HEAD_DIMS))}")
for i, (name, s) in enumerate(MAGNITUDES.items()):
    label = "  scores  :" if i == 0 else "           "
    print(f"{label} {name:<5} -> Q, K scaled by sigma = {s:<4}  (score std ~ {s * s:g})")
print(f"  plus max shape (B, H) = {MAX_BATCH_HEADS} at N = {', '.join(map(str, MAX_SHAPE_SEQ_LENS))}, "
      f"every D and score size")
print(f"  total   : {len(all_cases())} cases")
print("  'large' scores reach ~240, so exp() would overflow FP32 without the online (max-subtracted) softmax")
print()
EOF

python tests/fused/run_correctness.py --impl fused --kernel "$KERNEL" \
    --block-m "$TILE_M" --block-n "$TILE_N" --csv /tmp/demo_correctness.csv
status=$?

echo
if [ $status -eq 0 ]; then
    echo ">>> CORRECTNESS: ALL CASES PASSED"
else
    echo ">>> CORRECTNESS: SOME CASES FAILED (exit code $status)"
fi

# ------------------------------------------------------------------ 2. forward benchmark
banner "2. Forward benchmark: fused vs unfused, SDPA and TileGym"
python demo_forward.py
