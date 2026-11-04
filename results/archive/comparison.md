# Fused attention: version comparison

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition  
Configs: D in [64, 128], N in [128, 512, 2048, 8192]; baseline = v0_baseline  
Speedup = other_ms / version_ms (>1 = version faster). Averages over configs are geometric means.

## Summary per tile and version

| Tile | Version | Avg fused ms | Avg TFLOP/s | vs v0_baseline | vs unfused | vs SDPA |
|---|---|---|---|---|---|---|
| 64x64 | v0_baseline | 0.2551 | 98.96 | 1.00x | 3.21x | 1.02x |
| 64x64 | v1_heavyfirst | 0.2323 | 111.34 | 1.08x | 3.45x | 1.10x |
| 64x64 | v2_fusedInstructions | 0.2118 | 123.27 | 1.19x | 3.83x | 1.22x |
| 64x64 | v3_occupancy2 | 0.1806 | 144.61 | 1.29x | 4.15x | 1.32x |
| 64x128 | v0_baseline | 0.2738 | 92.63 | 1.00x | 3.25x | 0.98x |
| 64x128 | v1_heavyfirst | 0.2493 | 104.25 | 1.07x | 3.49x | 1.05x |
| 64x128 | v2_fusedInstructions | 0.2300 | 113.55 | 1.16x | 3.78x | 1.14x |
| 64x128 | v3_occupancy2 | 0.2164 | 118.18 | 1.16x | 3.75x | 1.13x |
| 128x64 | v0_baseline | 0.6641 | 51.74 | 1.00x | 1.90x | 0.48x |
| 128x64 | v1_heavyfirst | 0.5849 | 62.45 | 1.11x | 2.10x | 0.53x |
| 128x64 | v2_fusedInstructions | 0.6797 | 71.76 | 1.26x | 2.39x | 0.60x |
| 128x64 | v3_occupancy2 | 0.8398 | 67.42 | 1.16x | 2.21x | 0.56x |
| 128x128 | v0_baseline | 1.1046 | 27.80 | 1.00x | 1.71x | 0.32x |
| 128x128 | v1_heavyfirst | 1.0444 | 30.82 | 1.05x | 1.81x | 0.33x |
| 128x128 | v2_fusedInstructions | 1.1759 | 31.89 | 1.20x | 2.05x | 0.38x |
| 128x128 | v3_occupancy2 | 1.6996 | 26.81 | 1.02x | 1.76x | 0.32x |

## Tile 64x64: fused ms (speedup vs SDPA)

| D | N | SDPA ms | unfused ms | v0_baseline | v1_heavyfirst | v2_fusedInstructions | v3_occupancy2 | best |
|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0145 | 0.0285 | 0.0148 (0.98x) | 0.0147 (0.98x) | 0.0133 (1.09x) | 0.0133 (1.09x) | v2_fusedInstructions |
| 64 | 512 | 0.0237 | 0.0478 | 0.0191 (1.24x) | 0.0191 (1.24x) | 0.0174 (1.36x) | 0.0190 (1.25x) | v2_fusedInstructions |
| 64 | 2048 | 0.0855 | 0.3256 | 0.0783 (1.09x) | 0.0640 (1.34x) | 0.0539 (1.59x) | 0.0436 (1.96x) | v3_occupancy2 |
| 64 | 8192 | 0.5484 | 5.9928 | 0.6906 (0.79x) | 0.6323 (0.87x) | 0.5493 (1.00x) | 0.4469 (1.23x) | v3_occupancy2 |
| 128 | 128 | 0.0178 | 0.0293 | 0.0169 (1.05x) | 0.0169 (1.05x) | 0.0150 (1.18x) | 0.0148 (1.20x) | v3_occupancy2 |
| 128 | 512 | 0.0287 | 0.0526 | 0.0249 (1.15x) | 0.0249 (1.15x) | 0.0232 (1.24x) | 0.0230 (1.25x) | v3_occupancy2 |
| 128 | 2048 | 0.1162 | 0.3550 | 0.1132 (1.03x) | 0.0928 (1.25x) | 0.0865 (1.34x) | 0.0745 (1.56x) | v3_occupancy2 |
| 128 | 8192 | 1.0101 | 6.1380 | 1.0832 (0.93x) | 0.9940 (1.02x) | 0.9359 (1.08x) | 0.8097 (1.25x) | v3_occupancy2 |

## Tile 64x128: fused ms (speedup vs SDPA)

| D | N | SDPA ms | unfused ms | v0_baseline | v1_heavyfirst | v2_fusedInstructions | v3_occupancy2 | best |
|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0143 | 0.0314 | 0.0135 (1.06x) | 0.0139 (1.03x) | 0.0132 (1.08x) | 0.0134 (1.07x) | v2_fusedInstructions |
| 64 | 512 | 0.0241 | 0.0498 | 0.0211 (1.14x) | 0.0210 (1.15x) | 0.0200 (1.21x) | 0.0190 (1.27x) | v3_occupancy2 |
| 64 | 2048 | 0.0856 | 0.3422 | 0.0885 (0.97x) | 0.0721 (1.19x) | 0.0660 (1.30x) | 0.0567 (1.51x) | v3_occupancy2 |
| 64 | 8192 | 0.5473 | 6.0051 | 0.8322 (0.66x) | 0.7559 (0.72x) | 0.7106 (0.77x) | 0.5227 (1.05x) | v3_occupancy2 |
| 128 | 128 | 0.0178 | 0.0333 | 0.0169 (1.06x) | 0.0169 (1.06x) | 0.0149 (1.20x) | 0.0168 (1.06x) | v2_fusedInstructions |
| 128 | 512 | 0.0282 | 0.0553 | 0.0250 (1.13x) | 0.0245 (1.15x) | 0.0228 (1.23x) | 0.0291 (0.97x) | v2_fusedInstructions |
| 128 | 2048 | 0.1161 | 0.3762 | 0.1154 (1.01x) | 0.0948 (1.22x) | 0.0855 (1.36x) | 0.0966 (1.20x) | v2_fusedInstructions |
| 128 | 8192 | 1.0107 | 6.1166 | 1.0778 (0.94x) | 0.9954 (1.02x) | 0.9072 (1.11x) | 0.9768 (1.03x) | v2_fusedInstructions |

## Tile 128x64: fused ms (speedup vs SDPA)

| D | N | SDPA ms | unfused ms | v0_baseline | v1_heavyfirst | v2_fusedInstructions | v3_occupancy2 | best |
|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0143 | 0.0403 | 0.0169 (0.85x) | 0.0169 (0.85x) | 0.0150 (0.96x) | 0.0148 (0.97x) | v3_occupancy2 |
| 64 | 512 | 0.0236 | 0.0683 | 0.0273 (0.87x) | 0.0271 (0.87x) | 0.0231 (1.02x) | 0.0232 (1.02x) | v2_fusedInstructions |
| 64 | 2048 | 0.0854 | 0.3706 | 0.0947 (0.90x) | 0.0722 (1.18x) | 0.0570 (1.50x) | 0.0721 (1.19x) | v2_fusedInstructions |
| 64 | 8192 | 0.5481 | 6.2612 | 0.7771 (0.71x) | 0.6435 (0.85x) | 0.5021 (1.09x) | 0.4714 (1.16x) | v3_occupancy2 |
| 128 | 128 | 0.0178 | 0.0419 | 0.0599 (0.30x) | 0.0599 (0.30x) | 0.0372 (0.48x) | 0.0374 (0.48x) | v2_fusedInstructions |
| 128 | 512 | 0.0287 | 0.0784 | 0.1031 (0.28x) | 0.1043 (0.27x) | 0.0914 (0.31x) | 0.0970 (0.30x) | v2_fusedInstructions |
| 128 | 2048 | 0.1162 | 0.4117 | 0.4171 (0.28x) | 0.3187 (0.36x) | 0.3630 (0.32x) | 0.4180 (0.28x) | v1_heavyfirst |
| 128 | 8192 | 1.0109 | 6.4787 | 3.8163 (0.26x) | 3.4364 (0.29x) | 4.3486 (0.23x) | 5.5842 (0.18x) | v1_heavyfirst |

## Tile 128x128: fused ms (speedup vs SDPA)

| D | N | SDPA ms | unfused ms | v0_baseline | v1_heavyfirst | v2_fusedInstructions | v3_occupancy2 | best |
|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0144 | 0.0498 | 0.0251 (0.57x) | 0.0251 (0.57x) | 0.0207 (0.70x) | 0.0224 (0.64x) | v2_fusedInstructions |
| 64 | 512 | 0.0236 | 0.1051 | 0.0477 (0.50x) | 0.0494 (0.48x) | 0.0396 (0.60x) | 0.0457 (0.52x) | v2_fusedInstructions |
| 64 | 2048 | 0.0854 | 0.5419 | 0.1870 (0.46x) | 0.1559 (0.55x) | 0.1357 (0.63x) | 0.1665 (0.51x) | v2_fusedInstructions |
| 64 | 8192 | 0.5460 | 7.2991 | 1.7963 (0.30x) | 1.5997 (0.34x) | 1.4916 (0.37x) | 1.6228 (0.34x) | v2_fusedInstructions |
| 128 | 128 | 0.0178 | 0.0559 | 0.0619 (0.29x) | 0.0618 (0.29x) | 0.0369 (0.48x) | 0.0393 (0.45x) | v2_fusedInstructions |
| 128 | 512 | 0.0286 | 0.1244 | 0.1188 (0.24x) | 0.1213 (0.24x) | 0.1028 (0.28x) | 0.1110 (0.26x) | v2_fusedInstructions |
| 128 | 2048 | 0.1162 | 0.6434 | 0.5389 (0.22x) | 0.4690 (0.25x) | 0.5160 (0.23x) | 0.6008 (0.19x) | v1_heavyfirst |
| 128 | 8192 | 1.0114 | 7.7809 | 6.0611 (0.17x) | 5.8726 (0.17x) | 7.0636 (0.14x) | 10.9884 (0.09x) | v1_heavyfirst |

## Step-by-step change (geometric mean over all tiles and configs)

| Step | Speedup | Configs faster / total |
|---|---|---|
| v0_baseline -> v1_heavyfirst | 1.08x | 21 / 32 |
| v1_heavyfirst -> v2_fusedInstructions | 1.12x | 28 / 32 |
| v2_fusedInstructions -> v3_occupancy2 | 0.96x | 11 / 32 |

## Best version and tile per config

| D | N | Best | ms | vs SDPA | vs unfused (same tile) |
|---|---|---|---|---|---|
| 64 | 128 | v2_fusedInstructions 64x128 | 0.0132 | 1.08x | 2.38x |
| 64 | 512 | v2_fusedInstructions 64x64 | 0.0174 | 1.36x | 2.74x |
| 64 | 2048 | v3_occupancy2 64x64 | 0.0436 | 1.96x | 7.47x |
| 64 | 8192 | v3_occupancy2 64x64 | 0.4469 | 1.23x | 13.41x |
| 128 | 128 | v3_occupancy2 64x64 | 0.0148 | 1.20x | 1.98x |
| 128 | 512 | v2_fusedInstructions 64x128 | 0.0228 | 1.23x | 2.42x |
| 128 | 2048 | v3_occupancy2 64x64 | 0.0745 | 1.56x | 4.77x |
| 128 | 8192 | v3_occupancy2 64x64 | 0.8097 | 1.25x | 7.58x |
