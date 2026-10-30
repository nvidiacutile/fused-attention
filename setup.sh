#!/usr/bin/env bash
# One-command environment setup for a fresh Jarvislabs A30 instance.
#
#   git clone https://github.com/nvidiacutile/fused-attention.git
#   cd fused-attention
#   bash setup.sh
#   source .venv/bin/activate
#
# Creates .venv, installs the pinned requirements and writes the environment
# details to env.txt (used in README/report).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

echo "== Python environment =="
if [ ! -d "$VENV" ]; then
  if ! "$PYTHON" -m venv "$VENV"; then
    echo "python venv module missing, installing python3-venv" >&2
    apt-get update -qq && apt-get install -y -qq python3-venv
    "$PYTHON" -m venv "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "== Recording environment to env.txt =="
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu, driver, compute_cap: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader)"
  echo "python: $(python --version 2>&1)"
  python -c "import torch; print('torch:', torch.__version__, '| torch cuda:', torch.version.cuda)"
  echo "cuda-tile: $(pip show cuda-tile 2>/dev/null | awk '/^Version:/{print $2}')"
  echo "nvidia-cuda-nvcc: $(pip show nvidia-cuda-nvcc 2>/dev/null | awk '/^Version:/{print $2}')"
  if command -v ncu >/dev/null 2>&1; then
    echo "nsight compute: $(ncu --version | tail -n 1)"
  else
    echo "nsight compute: not found on PATH"
  fi
} > env.txt
cat env.txt

echo
echo "Setup complete. Activate with: source $VENV/bin/activate"
