# Re-run every correctness check and benchmark into results_replicate/, using the same folder layout
# and file names as results/, so the two trees can be compared file by file. results/ is not touched.
#
#   make                  everything (correctness first, then benchmarks)
#   make correctness      results_replicate/correctness/
#   make benchmarks       results_replicate/benchmarks/
#   make list             print every run without executing it
#   make compare          list files that exist in only one of results/ and results_replicate/
#   make -k ...           keep going after a failing run (a FAIL in a correctness run exits non-zero)
#   make OUT=some/dir     write somewhere else instead (OUT=results overwrites the committed results)
#
# Smaller pieces: correct-reference correct-unfused correct-fused correct-gqa correct-backward
#                 bench-fused bench-gqa bench-backward bench-tilegym
# Single runs:    make correct-fused/v3_occupancy2/64x128   make bench-fused/v0_baseline/32x64
#                 make bench-gqa/v0_baseline/64x64/Hkv2
#
# Run from the repo root. Needs a CUDA GPU and the .venv from setup.sh; bench-tilegym also needs
# NVIDIA TileGym installed. Runs one job at a time on purpose: parallel jobs on one GPU skew timings.

.NOTPARALLEL:
.DEFAULT_GOAL := all

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
OUT    ?= results_replicate
C      := $(OUT)/correctness
B      := $(OUT)/benchmarks

# ---------------------------------------------------------------- what exists in results/
FUSED_VERSIONS := v0_baseline v1_heavyfirst v2_fusedInstructions v3_occupancy2
TILES          := 64x64 64x128 128x64 128x128
V0_EXTRA_TILES := 32x64 64x32
GQA_VERSIONS   := v0_baseline
HKVS           := 1 2 4 8

FUSED_KERNEL = attention/fused/fused_kernels/$(1).py
GQA_KERNEL   = attention/gqa/fused_kernels/$(1).py

# Pieces of a run name such as bench-gqa/v0_baseline/64x128/Hkv2 (only valid inside a recipe)
V  = $(word 1,$(subst /, ,$*))
T  = $(word 2,$(subst /, ,$*))
H  = $(patsubst Hkv%,%,$(word 3,$(subst /, ,$*)))
TM = $(word 1,$(subst x, ,$(T)))
TN = $(word 2,$(subst x, ,$(T)))

# ---------------------------------------------------------------- run lists
CORRECT_FUSED := $(foreach v,$(FUSED_VERSIONS),$(foreach t,$(TILES),correct-fused/$(v)/$(t)))
CORRECT_GQA   := $(foreach v,$(GQA_VERSIONS),$(foreach t,$(TILES),correct-gqa/$(v)/$(t)))
BENCH_FUSED   := $(foreach v,$(FUSED_VERSIONS),$(foreach t,$(TILES),bench-fused/$(v)/$(t))) \
                 $(foreach t,$(V0_EXTRA_TILES),bench-fused/v0_baseline/$(t))
BENCH_GQA     := $(foreach v,$(GQA_VERSIONS),$(foreach t,$(TILES),$(foreach h,$(HKVS),bench-gqa/$(v)/$(t)/Hkv$(h))))

.PHONY: all correctness benchmarks list compare \
        correct-reference correct-unfused correct-fused correct-gqa correct-backward \
        bench-fused bench-gqa bench-backward bench-tilegym \
        $(CORRECT_FUSED) $(CORRECT_GQA) $(BENCH_FUSED) $(BENCH_GQA)

all: correctness benchmarks

# ---------------------------------------------------------------- $(OUT)/correctness/
correctness: correct-reference correct-unfused correct-fused correct-gqa correct-backward

correct-reference:
	$(PYTHON) tests/fused/run_correctness.py --impl reference \
	    --csv $(C)/reference/correctness_reference.csv
	$(PYTHON) tests/gqa/run_correctness_gqa.py --impl reference \
	    --csv $(C)/reference_gqa/correctness_reference_gqa.csv

correct-unfused:
	$(PYTHON) tests/fused/run_correctness.py --impl unfused \
	    --csv $(C)/unfused/correctness_unfused.csv
	$(PYTHON) tests/gqa/run_correctness_gqa.py --impl unfused \
	    --csv $(C)/unfused_gqa/correctness_unfused_gqa.csv

correct-fused: $(CORRECT_FUSED)
$(CORRECT_FUSED): correct-fused/%:
	$(PYTHON) tests/fused/run_correctness.py --impl fused --kernel $(call FUSED_KERNEL,$(V)) \
	    --block-m $(TM) --block-n $(TN) \
	    --csv $(C)/fused/$(V)/$(T)/correctness_fused_$(V)_$(T).csv

correct-gqa: $(CORRECT_GQA)
$(CORRECT_GQA): correct-gqa/%:
	$(PYTHON) tests/gqa/run_correctness_gqa.py --impl fused --kernel $(call GQA_KERNEL,$(V)) \
	    --block-m $(TM) --block-n $(TN) \
	    --csv $(C)/fused_gqa/$(V)/$(T)/correctness_fused_gqa_$(V)_$(T).csv

correct-backward:
	$(PYTHON) -m tests.backward.run_correctness_backward --tile-m 64 --tile-n 64 \
	    --out $(C)/backward/backward_64x64.csv

# ---------------------------------------------------------------- $(OUT)/benchmarks/
benchmarks: bench-fused bench-gqa bench-backward bench-tilegym

bench-fused: $(BENCH_FUSED)
$(BENCH_FUSED): bench-fused/%:
	$(PYTHON) benchmarks/benchmark_forward.py --kernel $(call FUSED_KERNEL,$(V)) \
	    --tile-m $(TM) --tile-n $(TN) \
	    --out         $(B)/fused/$(V)/$(T)/bench_forward_sep_$(V)_$(T).csv \
	    --summary-out $(B)/fused/$(V)/$(T)/bench_forward_agg_$(V)_$(T).csv \
	    --plot-out    $(B)/fused/$(V)/$(T)/bench_forward_plots_$(V)_$(T).png

bench-gqa: $(BENCH_GQA)
$(BENCH_GQA): bench-gqa/%:
	$(PYTHON) benchmarks/benchmark_forward_gqa.py --kernel $(call GQA_KERNEL,$(V)) \
	    --tile-m $(TM) --tile-n $(TN) --H-kv $(H) \
	    --out         $(B)/fused_gqa/$(V)/$(T)/Hkv$(H)/bench_forward_gqa_sep_$(V)_$(T)_Hkv$(H).csv \
	    --summary-out $(B)/fused_gqa/$(V)/$(T)/Hkv$(H)/bench_forward_gqa_agg_$(V)_$(T)_Hkv$(H).csv \
	    --plot-out    $(B)/fused_gqa/$(V)/$(T)/Hkv$(H)/bench_forward_gqa_plots_$(V)_$(T)_Hkv$(H).png

bench-backward:
	mkdir -p $(B)/backward
	$(PYTHON) -m benchmarks.benchmark_backward --tile-m 64 --tile-n 64 --out-dir $(B)/backward

bench-tilegym:
	mkdir -p $(B)/fused/tilegym
	$(PYTHON) benchmarks/benchmark_tilegym.py --kernel $(call FUSED_KERNEL,v3_occupancy2) \
	    --out-dir $(B)/fused/tilegym

# ---------------------------------------------------------------- helpers
list:
	@$(MAKE) --no-print-directory -n all | grep -v '^make'

# Files present in only one of the two trees (README.md files are hand-written, not generated)
compare:
	@diff <(cd results && find correctness benchmarks -type f ! -name README.md | sort) \
	      <(cd $(OUT) && find correctness benchmarks -type f | sort) \
	  && echo "same file list in results/ and $(OUT)/"

compare: SHELL := /bin/bash
