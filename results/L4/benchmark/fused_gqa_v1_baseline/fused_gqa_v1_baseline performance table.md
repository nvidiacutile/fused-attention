# fused_gqa_v1_baseline_performance_table

GQA forward, kernel `attention/fused_gqa.py` (v3_occupancy2 + GQA). B=2, H=8 query heads, H_kv ∈ {1, 2, 4, 8} K/V heads, NVIDIA L4.
Averages are over D ∈ {32, 64, 128}, N ∈ {128, 512, 2048, 8192}; speedup vs unfused is the geometric mean of per-config ratios. **Bold** = lowest fused latency for that H_kv.

## H_kv summary (best tile per H_kv)

| H_kv | Group (H/H_kv) | Best tile | Fused ms | Fused TFLOPS | SDPA TFLOPS | % of SDPA TFLOPS | Speedup vs unfused |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 64×64 | 0.877 | 30.14 | 31.84 | 94.7% | 5.40× |
| 2 | 4 | 64×64 | 0.881 | 29.60 | 30.60 | 96.7% | 5.34× |
| 4 | 2 | 64×64 | 0.899 | 29.06 | 30.02 | 96.8% | 5.27× |
| 8 | 1 | 64×64 | 0.917 | 29.95 | 30.23 | 99.1% | 5.40× |

## Tile summary, H_kv=1 (group 8)

| Tile (M×N) | Fused ms | Fused TFLOPS | Unfused TFLOPS | SDPA TFLOPS | Speedup vs unfused | % of SDPA TFLOPS |
|---|---|---|---|---|---|---|
| **64×64** | **0.877** | **30.14** | 4.20 | 31.84 | **5.40×** | **94.7%** |
| 64×128 | 0.950 | 29.11 | 3.79 | 30.60 | 5.79× | 95.1% |
| 128×64 | 0.949 | 28.41 | 3.33 | 30.47 | 6.27× | 93.2% |
| 128×128 | 1.334 | 27.13 | 3.05 | 30.73 | 6.60× | 88.3% |

## Tile summary, H_kv=2 (group 4)

| Tile (M×N) | Fused ms | Fused TFLOPS | Unfused TFLOPS | SDPA TFLOPS | Speedup vs unfused | % of SDPA TFLOPS |
|---|---|---|---|---|---|---|
| **64×64** | **0.881** | **29.60** | 4.19 | 30.60 | **5.34×** | **96.7%** |
| 64×128 | 0.956 | 28.81 | 3.78 | 31.01 | 5.74× | 92.9% |
| 128×64 | 0.941 | 28.85 | 3.32 | 30.26 | 6.29× | 95.4% |
| 128×128 | 1.367 | 25.50 | 3.04 | 30.49 | 6.40× | 83.6% |

## Tile summary, H_kv=4 (group 2)

| Tile (M×N) | Fused ms | Fused TFLOPS | Unfused TFLOPS | SDPA TFLOPS | Speedup vs unfused | % of SDPA TFLOPS |
|---|---|---|---|---|---|---|
| **64×64** | **0.899** | **29.06** | 4.17 | 30.02 | **5.27×** | **96.8%** |
| 64×128 | 0.962 | 28.85 | 3.78 | 30.58 | 5.82× | 94.3% |
| 128×64 | 0.967 | 27.84 | 3.31 | 30.52 | 6.21× | 91.2% |
| 128×128 | 1.345 | 25.77 | 3.03 | 30.33 | 6.46× | 85.0% |

## Tile summary, H_kv=8 (group 1)

| Tile (M×N) | Fused ms | Fused TFLOPS | Unfused TFLOPS | SDPA TFLOPS | Speedup vs unfused | % of SDPA TFLOPS |
|---|---|---|---|---|---|---|
| **64×64** | **0.917** | **29.95** | 4.17 | 30.23 | **5.40×** | **99.1%** |
| 64×128 | 1.043 | 27.99 | 3.77 | 30.36 | 5.68× | 92.2% |
| 128×64 | 0.969 | 28.41 | 3.30 | 29.61 | 6.28× | 95.9% |
| 128×128 | 1.386 | 25.44 | 3.02 | 30.56 | 6.41× | 83.2% |

## Fused ms by tile × H_kv (averaged over D, N)

| Tile \ H_kv | H_kv=1 | H_kv=2 | H_kv=4 | H_kv=8 |
|---|---|---|---|---|
| 64×64 | **0.877** | **0.881** | **0.899** | **0.917** |
| 64×128 | 0.950 | 0.956 | 0.962 | 1.043 |
| 128×64 | 0.949 | 0.941 | 0.967 | 0.969 |
| 128×128 | 1.334 | 1.367 | 1.345 | 1.386 |

## Best tile per N × D, H_kv=1 (lowest fused latency)

| N \ D | D=32 | D=64 | D=128 |
|---|---|---|---|
| N=128 | 128×128 | 64×64 | 64×128 |
| N=512 | 64×128 | 64×64 | 64×64 |
| N=2048 | 128×64 | 64×64 | 64×64 |
| N=8192 | 128×64 | 128×128 | 64×64 |

## Best tile per N × D, H_kv=2 (lowest fused latency)

| N \ D | D=32 | D=64 | D=128 |
|---|---|---|---|
| N=128 | 64×64 | 64×64 | 64×64 |
| N=512 | 64×128 | 64×64 | 64×64 |
| N=2048 | 128×64 | 128×128 | 64×64 |
| N=8192 | 128×64 | 64×64 | 64×64 |

## Best tile per N × D, H_kv=4 (lowest fused latency)

| N \ D | D=32 | D=64 | D=128 |
|---|---|---|---|
| N=128 | 64×128 | 64×64 | 64×128 |
| N=512 | 64×128 | 64×64 | 64×128 |
| N=2048 | 128×64 | 128×128 | 64×64 |
| N=8192 | 64×128 | 128×128 | 64×64 |

## Best tile per N × D, H_kv=8 (lowest fused latency)

| N \ D | D=32 | D=64 | D=128 |
|---|---|---|---|
| N=128 | 64×128 | 64×64 | 64×128 |
| N=512 | 64×128 | 64×64 | 64×64 |
| N=2048 | 128×64 | 128×128 | 64×64 |
| N=8192 | 128×64 | 64×64 | 64×64 |
