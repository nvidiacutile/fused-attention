# fused-attention

Causal scaled dot-product attention in NVIDIA cuTile: a fused FlashAttention-style forward kernel with online softmax, extended to grouped-query attention (GQA/MQA) and a recompute-based backward pass with PyTorch autograd support, tested and benchmarked against an unfused cuTile baseline and PyTorch SDPA.

## Status

| Task | What | Owner | File(s) |
|---|---|---|---|
| T1 | Environment setup | Both | `README.md`, `requirements.txt` |
| T2 | PyTorch causal SDPA reference (FP16 in, FP32 intermediates) | Nishitha | `attention/reference.py` |
| T3 | Unfused cuTile baseline (QKᵀ → softmax → PV) | Nishitha | `attention/unfused.py` |
| T4 | `run_attention.py` CLI | Harshith | `run_attention.py`, `attention/validation.py` |
| T5 | Correctness tests | Harshith | `tests/` |
| T6 | Fused causal attention forward kernel (FlashAttention-2 style) | Nishitha | `attention/fused.py` |
| T7 | Forward benchmark + analysis (fused vs unfused vs SDPA) | Nishitha | `benchmarks/bench_forward.py`, `benchmarks/benchmark_analysis.py` |

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA A30 (Ampere, 24 GB), Jarvislabs |
| Compute capability | 8.0 |
| NVIDIA driver | 580.126.20 (supports CUDA 13.0) |
| Python | 3.10.20 |
| PyTorch | 2.14.1+cu130 (CUDA 13.0) |
| cuda-tile (cuTile) | 1.6.0 |
| nvcc (`nvidia-cuda-nvcc`) | 13.4.92 |
| Nsight Compute | 2024.3.2.0 (build 34861637) |

## Setup

Jarvislabs instances start empty, so after logging in to the A30 instance (Jarvislabs dashboard → Instances → **VS Code**, password from the **Password** button) open a terminal and do the steps below each time.

### 1. GitHub access 

The repo is private, so cloning and pushing need a personal access token (PAT).

On each new instance:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global credential.helper store   # remember the token until the instance is wiped
```

Alternative if `gh` is installed (`which gh`): run `gh auth login` → GitHub.com → HTTPS → Login with a web browser, then enter the one-time code at github.com/login/device.

### 2. Clone and set up

```bash
git clone https://github.com/nvidiacutile/fused-attention.git
# Username: nvidiacutile
# Password: paste the token (not the GitHub password)
cd fused-attention
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv`, installs the pinned `requirements.txt` and records the environment to `env.txt`.

### 3. Before pausing the instance

```bash
git add -A && git commit -m "..." && git push
```

Anything not pushed is lost when the instance is paused or destroyed. The instance password also changes after pause/resume.

## Usage

```bash
python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile
```

- **Input:** `INPUT.npz` with contiguous float16 arrays `Q` `[B, H, N, D]`, `K` `[B, H, N, D]`, `V` `[B, H, N, D]` (Q, K and V share the head dim D).
- **Limits:** 1 ≤ B ≤ 16, 1 ≤ H ≤ 32, 1 ≤ N ≤ 8192, D ∈ {32, 64, 128}, N divisible by `--block-m` and `--block-n` (default 64).
- **Output:** float16 array of shape `[B, H, N, D]`.
- **Backends:** `cutile` (fused kernel), `unfused` (baseline), `reference` (PyTorch), `cutedsl` (optional track, not implemented).
- **Exit status:** 0 success, 2 unsupported input (with an error message), 1 runtime failure.

## Tests

### Correctness

```bash
python tests/run_correctness.py --impl unfused                 # or --impl fused
python tests/run_correctness.py --impl fused --block-m 128 --block-n 64
python tests/run_correctness.py --impl fused --filter N512     # subset by case id
```

Runs the chosen kernel and the FP32 PyTorch reference on 84 fixed-seed cases from
`tests/cases.py`:

- D ∈ {32, 64, 128} × N ∈ {512, 4096, 8192} × tiny/large score magnitudes × (B, H) ∈ {(1, 1), (2, 8), (3, 5)}
- plus the maximum shape (B, H) = (16, 32) at N = 512 for each D and magnitude

Tiny scores (σ = 0.03, score std ≈ 1e-3) give a near-uniform softmax; large scores
(σ = 7, max |score| ≈ 240) overflow `exp` in FP32 unless the softmax subtracts the running max.
A case fails on any NaN/Inf, or if any element violates `|out − ref| ≤ 2e-2 + 2e-2·|ref|`.
Prints max absolute/relative error and PASS/FAIL per case, writes
`results/correctness_<impl>.csv`, and exits 0 only if every case passes.

### Unfused and reference

```bash
source .venv/bin/activate
python tests/run_correctness.py --impl reference    # harness self-check: reference vs itself
python tests/run_correctness.py --impl unfused      # unfused cuTile baseline vs reference
```

Each runs all 84 cases and should end with `CORRECT: all 84 cases passed`, writing
`results/correctness_reference.csv` and `results/correctness_unfused.csv`. Add
`--filter N512` (or any part of a case id) to run a subset. The unfused baseline always
uses 64×64 tiles, so `--block-m` / `--block-n` only affect validation for it.

## Fused kernel

`attention/fused.py` provides `fused_attention(q, k, v, tile_m=64, tile_n=64)`, used by the `cutile` backend of `run_attention.py` and by `--impl fused` in the tests.

- One block per (query tile, batch·head): grid `(N / tile_m, B·H)`.
- Each block loads its Q tile once, then loops over K/V tiles keeping the scores, the softmax tile and the output accumulator on-chip; S and P are never written to global memory.
- Online softmax with a running row max `m` and row sum `l`; the output is divided by `l` once at the end.
- Causal tile skipping: K/V tiles entirely in the future of the Q tile are not loaded; only the diagonal tile is masked.
- FP16 inputs, FP32 accumulation (`ct.mma`), FP16 output.
- Currently requires `tile_m == tile_n`.

Status: passes all 60 correctness cases (`python tests/run_correctness.py --impl fused`).

## Benchmarking

```bash
source .venv/bin/activate
python -m benchmarks.bench_forward          # time fused, unfused and SDPA → results/bench_forward.csv
python -m benchmarks.benchmark_analysis     # tables, observations, plot from that CSV
```

Run both from the repo root with `-m` so that `attention/` is importable.

`bench_forward` options:

| Flag | Default | Meaning |
|---|---|---|
| `--B`, `--H` | 2, 8 | batch size and number of heads |
| `--tile-m`, `--tile-n` | 64, 64 | tile sizes for fused and unfused |
| `--impls` | `fused,unfused,sdpa` | comma-separated implementations to time |
| `--out` | `results/bench_forward.csv` | output CSV |

Problem sizes: D ∈ {64, 128} × N ∈ {512, 2048, 8192}, random FP16 inputs, causal.

Method: one first call timed separately (includes JIT compile), 10 untimed warm-up runs, then the median of 30 runs timed with CUDA events. Throughput uses the causal FLOP count 4·B·H·N²·D / 2. SDPA is `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)` in FP16 and has no user tile size.

`benchmark_analysis` options: `--csv` (default `results/bench_forward.csv`), `--peak` (GPU FP16 peak TFLOP/s for the %-of-peak column), `--out-md`, `--out-png`. It writes:

- `results/bench_forward_summary.md`: latency table, speedups, global-memory traffic avoided by fusion, first-call times, observations
- `results/bench_forward.png`: latency vs N per implementation (needs `matplotlib`)

To compare tile sizes, write each run to its own CSV:

```bash
python -m benchmarks.bench_forward --tile-m 128 --tile-n 128 --out results/bench_forward_128.csv
python -m benchmarks.benchmark_analysis --csv results/bench_forward_128.csv \
    --out-md results/bench_forward_128_summary.md --out-png results/bench_forward_128.png
```

### Forward results (B = 2, H = 8, 64×64 tiles)

Run on 2026-10-29 at about 16:30 IST.

| Component | Value |
|---|---|
| GPU | NVIDIA L4 (Ada Lovelace, 24 GB), Jarvislabs |
| Compute capability | 8.9 |
| NVIDIA driver | 580.126.20 (supports CUDA 13.0) |
| Python | 3.10.20 |
| PyTorch | 2.14.1+cu130 (CUDA 13.0) |
| cuda-tile (cuTile) | 1.6.0 |
| Inputs | random FP16, seed 0, contiguous `[B, H, N, D]` |
| Raw data | `results/bench_forward.csv` |

Fused kernel at commit `967838a`.

| D | N | unfused ms | fused ms | SDPA ms | fused vs unfused | fused vs SDPA |
|---|---|---|---|---|---|---|
| 64 | 512 | 0.221 | 0.055 | 0.051 | 4.0× faster | 1.08× slower |
| 64 | 2048 | 1.958 | 0.218 | 0.166 | 9.0× faster | 1.31× slower |
| 64 | 8192 | 19.193 | 2.216 | 0.986 | 8.7× faster | 2.25× slower |
| 128 | 512 | 0.172 | 0.056 | 0.066 | 3.1× faster | 1.16× faster |
| 128 | 2048 | 1.597 | 0.234 | 0.193 | 6.8× faster | 1.21× slower |
| 128 | 8192 | 21.264 | 2.629 | 1.961 | 8.1× faster | 1.34× slower |

- Fusion gives 3–9× over the unfused baseline, growing with N: at N = 8192 the unfused kernels move about 6 GB of S and P through global memory.
- The fused kernel beats SDPA at D = 128, N = 512 and is 1.08–2.25× slower elsewhere; the largest gap is D = 64, N = 8192, the first target for profiling.
- At N = 512 all runs are around 0.05 ms, so launch overhead dominates there.

## Progress log

Important runs, results and commits are recorded with timestamps in [`docs/progress_log.md`](docs/progress_log.md).
