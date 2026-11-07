# GQA profiling

Performance runs for the GQA/MQA fused kernel (`attention/gqa/fused_gqa.py`).

This folder is laid out and used the same way as `profiling/`. Follow the instructions in [`../profiling/README.md`](../profiling/README.md), replacing `profiling` with `gqa_profiling` in the commands and adding `--H-kv` for the number of K/V heads (default 2):

    sudo env "PATH=$PATH" python -m gqa_profiling.track_run --ncu --H-kv 2 \
        --kernel gqa_profiling/1_optimizations/v0_baseline/v0_baseline.py --note "..."

- `1_optimizations/`: kernels that were kept, each with its run(s)
- `2_rejected/`: kernels that did not make it faster (see `2_rejected/README.md`)
- `RUNS.md`, `runs.csv`: every run, as in `profiling/`
