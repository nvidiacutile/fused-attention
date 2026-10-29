# Forward benchmark summary

B=2, H=8, tile=64x64 (fused/unfused), FP16 inputs, causal. Peak used for %: 121 TFLOP/s.

## Latency (median ms) and throughput

| D | N | unfused ms | fused ms | SDPA ms | fused TFLOP/s | fused % peak | fused vs unfused | fused vs SDPA |
|---|---|---|---|---|---|---|---|---|
| 64 | 512 | 0.221 | 0.055 | 0.051 | 9.71 | 8.0% | 4.00x faster | 1.08x slower |
| 64 | 2048 | 1.958 | 0.218 | 0.166 | 39.38 | 32.5% | 8.98x faster | 1.31x slower |
| 64 | 8192 | 19.193 | 2.216 | 0.986 | 62.01 | 51.2% | 8.66x faster | 2.25x slower |
| 128 | 512 | 0.172 | 0.056 | 0.066 | 19.07 | 15.8% | 3.06x faster | 1.16x faster |
| 128 | 2048 | 1.597 | 0.234 | 0.193 | 73.58 | 60.8% | 6.84x faster | 1.21x slower |
| 128 | 8192 | 21.264 | 2.629 | 1.961 | 104.57 | 86.4% | 8.09x faster | 1.34x slower |

## Global-memory traffic avoided by fusion

Unfused writes S (FP32) and P (FP16), each [B,H,N,N], to global memory and reads them back. Fused keeps them on-chip.

| N | S+P size in HBM (unfused) | Q+K+V+O size |
|---|---|---|
| 512 | 24.0 MB | 8.0 MB (D=128) |
| 2048 | 384.0 MB | 32.0 MB (D=128) |
| 8192 | 6.0 GB | 128.0 MB (D=128) |

## First-call time (includes JIT compile; not part of latency above)

- fused: max 202 ms, min 0 ms
- unfused: max 120 ms, min 2 ms
- sdpa: max 11 ms, min 1 ms

## Observations

- D=64: fused-vs-unfused speedup grows with N (4.00x at N=512 -> 8.66x at N=8192). Expected: unfused traffic grows as N^2, so fusion helps more at long sequences.
- D=128: fused-vs-unfused speedup grows with N (3.06x at N=512 -> 8.09x at N=8192). Expected: unfused traffic grows as N^2, so fusion helps more at long sequences.
- Fused is slower than SDPA in 5/6 sizes (worst 2.25x at D=64, N=8192). This is the gap the optimization phase targets.
- Fused matches or beats SDPA at: D=128 N=512.
- Best fused utilization: 86.4% of peak. High: close to hardware limits.
- WARNING: 2 result(s) exceed the assumed peak of 121 TFLOP/s (e.g. sdpa at D=64, N=8192). The peak value is wrong for this GPU/clock, so treat '% peak' as unreliable; compare against SDPA instead, or pass --peak with the correct number.

