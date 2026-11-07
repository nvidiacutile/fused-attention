This is the first rough version of profiling - v0

It incudes equal tile only baseline implementation (v0 in kernels) and the subsequent optimizations
  Run 1: L4 baseline (old v1_baseline.py)
  Run 2: same on the RTX PRO 6000
  Run 3: occupancy=2
  Run 4: occupancy 2 + heavy-first 
  Runs 5–7: baselines after adding unequal-tile support (tiles/unequalTiles series)
  
For the latest code refer : profiling/
