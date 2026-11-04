# TODO: RTX 6000 PRO benchmark runs

Replicate the earlier benchmark outputs on the RTX 6000 PRO machine. Run everything from the repo root
(fused-attention/), on the RTX 6000 PRO GPU, and send all outputs into this folder.

- [ ] 1. Create the output folder: results/RTX_6000_PRO/fused/
- [ ] 2. Forward benchmarks for v0_baseline, v1_heavyfirst, v2_fusedInstructions and v3_occupancy2
- [ ] 3. TileGym comparison (needs TileGym installed)
- [ ] 4. Check the outputs (checklist at the end)

## 1. Create the folder

    mkdir -p results/RTX_6000_PRO/fused

Target layout when everything is done:

    results/RTX_6000_PRO/fused/
      tilegym/                      <- benchmarks/benchmark_tilegym.py
      v0_baseline/<tile>/           <- benchmarks/benchmark_forward.py
      v1_heavyfirst/<tile>/
      v2_fusedInstructions/<tile>/
      v3_occupancy2/<tile>/

## 2. Forward benchmarks: benchmarks/benchmark_forward.py

Kernels are in attention/fused/fused_kernels/. Run each version at tiles 64x64, 64x128, 128x64 and 128x128
(the four tiles the earlier runs in temp_results/ share; v0_baseline also had 32x64 and 64x32, which are optional).

The script's default output folder is results/benchmark/<tag>/<tile>/, so pass explicit paths with
--out, --summary-out and --plot-out to land in the RTX_6000_PRO folder:

    OUT=results/RTX_6000_PRO/fused
    for v in v0_baseline v1_heavyfirst v2_fusedInstructions v3_occupancy2; do
      for t in "64 64" "64 128" "128 64" "128 128"; do
        set -- $t; tile=$1x$2; d=$OUT/$v/$tile
        python benchmarks/benchmark_forward.py \
          --kernel attention/fused/fused_kernels/$v.py --tag $v --tile-m $1 --tile-n $2 \
          --out $d/bench_forward_sep_"$v"_"$tile".csv \
          --summary-out $d/bench_forward_agg_"$v"_"$tile".csv \
          --plot-out $d/bench_forward_plots_"$v"_"$tile".png
      done
    done

Each run uses B=2, H=8, D in (64, 128), N in (128, 512, 2048, 8192), and compares fused, unfused and SDPA.
Each (version, tile) folder should end up with three files: the per-config CSV (bench_forward_sep_*),
the aggregate CSV (bench_forward_agg_*) and the plot PNG (bench_forward_plots_*).

## 3. TileGym comparison: benchmarks/benchmark_tilegym.py

Compares our v3_occupancy2 kernel (64x64 tiles, the default --kernel) against TileGym tile_fmha and SDPA,
with the same sweep, B=2, H=8, 10 warm-up and 30 timed runs. TileGym must be importable
(from tilegym.ops.cutile.attention import tile_fmha); install it first if the script reports it cannot import it.

    python benchmarks/benchmark_tilegym.py --out-dir results/RTX_6000_PRO/fused/tilegym

Outputs in that folder: bench_tilegym_sep_*.csv (per config), bench_tilegym_agg_*.csv (per N / per D / overall)
and bench_tilegym_plots_*.png. Speedups reported: ours vs TileGym, ours vs SDPA, TileGym vs SDPA (>1 = first-named faster).

## 4. Checklist when done

- [ ] nvidia-smi shows the RTX 6000 PRO, and the gpu column in the CSVs says so
- [ ] 16 forward runs finished (4 versions x 4 tiles), each folder has its 3 files
- [ ] results/RTX_6000_PRO/fused/tilegym/ has its 3 files
- [ ] No stray outputs were written to results/benchmark/ by mistake
