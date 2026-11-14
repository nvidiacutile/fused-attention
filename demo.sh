#!/usr/bin/env bash
# Demo: correctness and forward benchmark, for the fused kernel and for GQA.
# Run from the repo root, with demo_forward.py next to it:   bash demo.sh
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null

KERNEL=attention/fused/fused_kernels/v3_occupancy2.py
GQA_KERNEL=attention/gqa/fused_kernels/v0_baseline.py     # GQA version of v3 (same as attention/gqa/fused_gqa.py)
GQA_HKV=2                                                 # K/V heads used in the GQA benchmark
TILE_M=64
TILE_N=64

banner() { echo; echo "==================== $1 ===================="; echo; }
verdict() {
    echo
    if [ "$1" -eq 0 ]; then echo ">>> $2: ALL CASES PASSED"
    else echo ">>> $2: SOME CASES FAILED (exit code $1)"; fi
}

# ------------------------------------------------------------------ 1. fused correctness
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
verdict $? "FUSED CORRECTNESS"

# ------------------------------------------------------------------ 2. fused benchmark
banner "2. Forward benchmark: fused vs unfused, SDPA and TileGym"
python demo_forward.py

# ------------------------------------------------------------------ 3. GQA correctness
banner "3. Correctness: GQA kernel vs FP32 PyTorch reference"

python - <<'EOF'
from tests.gqa.cases_gqa import BATCH, NUM_Q_HEADS, NUM_KV_HEADS, SEQ_LENS, HEAD_DIMS, MAGNITUDES, all_cases
print("Test grid we iterate over:")
print(f"  B       : {BATCH}")
print(f"  H       : {NUM_Q_HEADS} query heads")
print(f"  H_kv    : {', '.join(map(str, NUM_KV_HEADS))} K/V heads  "
      f"(1 = multi-query attention, {NUM_Q_HEADS} = plain multi-head attention)")
print(f"  N       : {', '.join(map(str, SEQ_LENS))}")
print(f"  D       : {', '.join(map(str, HEAD_DIMS))}")
for i, (name, s) in enumerate(MAGNITUDES.items()):
    label = "  scores  :" if i == 0 else "           "
    print(f"{label} {name:<5} -> Q, K scaled by sigma = {s:<4}  (score std ~ {s * s:g})")
print(f"  total   : {len(all_cases())} cases")
print()
EOF

python tests/gqa/run_correctness_gqa.py --impl fused --kernel "$GQA_KERNEL" \
    --block-m "$TILE_M" --block-n "$TILE_N" --csv /tmp/demo_correctness_gqa.csv
verdict $? "GQA CORRECTNESS"

# ------------------------------------------------------------------ 4. GQA benchmark
banner "4. GQA forward benchmark: fused vs unfused, SDPA and TileGym (H_kv = $GQA_HKV)"
python demo_forward.py --hkv "$GQA_HKV"
