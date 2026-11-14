"""Demo: summary table of the forward, GQA and backward benchmarks (written by demo.sh).
    python demo_summary.py FWD.json GQA.json BWD.json FWD_OK GQA_OK BWD_OK [H_KV]
FWD_OK etc. are "PASS" or "FAIL" from the correctness steps.
"""
import json
import sys

fwd, gqa, bwd = (json.load(open(p)) for p in sys.argv[1:4])
ok = sys.argv[4:7]
hkv = sys.argv[7] if len(sys.argv) > 7 else "2"


def x(v):
    return f"{v:.2f}x" if v else "-"


rows = [
    ("Forward (MHA)", "unfused cuTile", fwd, ok[0], "84 cases"),
    (f"GQA (H_kv = {hkv})", "unfused cuTile", gqa, ok[1], "96 cases"),
    ("Backward (fwd+bwd)", "naive PyTorch", bwd, ok[2], "7 cases"),
]

hdr = (f"{'Benchmark':<20} | {'baseline':<15} {'vs baseline':>11} | {'vs SDPA':>7} {'SDPA range':>13} | "
       f"{'vs TileGym':>10} | {'max err':>8} | {'correctness':<16}")
print(hdr)
print("-" * len(hdr))
for name, base, r, verdict, n in rows:
    rng = f"{r['min_vs_sdpa']:.2f}-{r['max_vs_sdpa']:.2f}x"
    print(f"{name:<20} | {base:<15} {x(r['vs_baseline']):>11} | {x(r['vs_sdpa']):>7} {rng:>13} | "
          f"{x(r['vs_tilegym']):>10} | {r['max_err']:8.1e} | {verdict} ({n})")
print("-" * len(hdr))
print("Speedups are geometric means over 8 configs (D in {64, 128} x N in {128, 512, 2048, 8192}), "
      "> 1 means ours is faster.")
print("max err: forward and GQA = max |ours - FP32 ref|; backward = max relative error of dQ, dK, dV.")
print(f"Backward: up to {bwd['max_vs_baseline']:.1f}x faster and up to {bwd['max_mem_ratio']:.0f}x less "
      f"memory than naive PyTorch (largest at N = 8192).")
