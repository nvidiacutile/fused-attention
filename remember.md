# Remember

- Future optimization (fused.py): split the KV loop into an unmasked loop `range(mask_start)` and a masked loop `range(mask_start, num_kv_tiles)` to remove the per-iteration `if j >= mask_start` branch; benchmark against fused_v1_baseline_64x64.
