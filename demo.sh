#!/usr/bin/env bash
# Whole demo in one command. Run from the repo root (with demo_speed.py next to it):
#     bash demo.sh
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null

step() { echo; echo "=================== $1 ==================="; }

[ -f demo.npz ] || python -c "import numpy as np; r=np.random.randn; np.savez('demo.npz', Q=r(2,8,2048,64).astype('f2'), K=r(2,8,2048,64).astype('f2'), V=r(2,8,2048,64).astype('f2'))"

step "1. CLI with valid input"
python run_attention.py --input demo.npz --output out.npy --verbose
echo "exit code $?"

step "2. CLI with invalid input (N=2048 not divisible by 48)"
python run_attention.py --input demo.npz --output out.npy --block-m 48
echo "exit code $?"

step "3. Correctness: v3, 84 cases vs FP32 reference"
make correct-fused/v3_occupancy2/64x64

step "4. Speed (vs unfused, SDPA, TileGym), GQA and backward"
python demo_speed.py
