# ===========================================================================
#  Entry points for everything in this repository.
#  @tandonmitul27  --  AUTHORED FILE (new)
# ===========================================================================
#
#  WHY THIS FILE EXISTS
#      Every gem5 invocation here needs the same three things right: the
#      working directory must be the gem5 root (device-config paths are
#      relative to it), LD_LIBRARY_PATH must find libdramsim3.so, and the
#      right harness flags must be passed.  Getting any of them wrong
#      produces a plausible-looking but wrong number rather than an
#      error -- which is exactly the failure mode this file removes.
#
#      Run `make help` for the list.
# ===========================================================================

SHELL   := /bin/bash
ROOT    := $(shell pwd)
GEM5    := $(ROOT)/gem5
DRAMSIM := $(GEM5)/ext/dramsim3/DRAMsim3
BIN     := $(GEM5)/build/NULL/gem5.opt
PY      ?= python3
JOBS    ?= $(shell nproc 2>/dev/null || echo 4)

# libdramsim3.so lives beside the DRAMsim3 sources; conda users also need
# their own lib dir on the path for zlib/protobuf.
export LD_LIBRARY_PATH := $(HOME)/miniconda3/lib:$(DRAMSIM):$(LD_LIBRARY_PATH)

# Harnesses are invoked from the gem5 root: DRAMsim3 config paths inside
# them are relative to it.
RUN = cd $(GEM5) && $(BIN)

.PHONY: help setup check check-fast params addrmap energy bw-channel \
        bw-stack cxl-latency cxl-bandwidth clean

help:
	@echo ""
	@echo "  make setup          build gem5 + DRAMsim3 with our patches/configs"
	@echo ""
	@echo "  make check          full validation suite   (~5 min)"
	@echo "  make check-fast     skips full-stack runs   (~2 min)"
	@echo ""
	@echo "  make params         print every parameter with its provenance"
	@echo "  make addrmap        print the static HBM/CXL map for each model"
	@echo ""
	@echo "  make bw-channel     HBM3 single-channel bandwidth"
	@echo "  make bw-stack       HBM3 full-stack bandwidth (16 channels)"
	@echo "  make cxl-latency    CXL added latency, ASIC and FPGA devices"
	@echo "  make cxl-bandwidth  effective bandwidth of the CXL 2.0 x16 link"
	@echo "  make energy         pJ/bit for HBM3, HBM3E and the DDR5 media"
	@echo ""
	@echo "  MODEL=<tag> HBM=<gib> selects the model for addrmap"
	@echo ""

setup:
	./setup.sh --jobs $(JOBS)

$(BIN):
	@echo "gem5 is not built. Run 'make setup' first." && exit 1

# --- validation ------------------------------------------------------------
check: $(BIN)
	$(PY) sim/check.py

check-fast: $(BIN)
	$(PY) sim/check.py --fast

# --- reference tables ------------------------------------------------------
params:
	$(PY) sim/system_params.py

MODEL ?= Mixtral-8x7B
HBM   ?= 80
addrmap:
	$(PY) mapping/address_map.py --tag $(MODEL) --hbm-gib $(HBM)

# --- individual measurements ----------------------------------------------
bw-channel: $(BIN)
	$(RUN) --outdir=$(ROOT)/out/bw-channel ../sim/configs/smoke_hbm.py \
	  --config HBM3_16Gb_x64_1ch --mem-size 1GB --window-ns 20000 --direct
	@awk '/bwRead::total/ {s+=$$2} END {printf "\n  HBM3 single channel: %.1f GB/s\n\n", s/1e9}' \
	  $(ROOT)/out/bw-channel/stats.txt

bw-stack: $(BIN)
	$(RUN) --outdir=$(ROOT)/out/bw-stack ../sim/configs/smoke_hbm.py \
	  --config HBM3_16Gb_x64_1ch --mem-size 2GB --window-ns 20000 \
	  --pairs 16 --sys-ghz 8.0
	@awk '/bwRead::total/ {s+=$$2} END {printf "\n  HBM3 stack (16 ch): %.1f GB/s   [H100 datasheet implies 670]\n\n", s/1e9}' \
	  $(ROOT)/out/bw-stack/stats.txt

cxl-latency: $(BIN)
	@for m in "local --no-link" "asic --link-gbps 63" \
	          "fpga --link-gbps 63 --dev-proto-ns 60"; do \
	  set -- $$m; name=$$1; shift; \
	  $(RUN) --outdir=$(ROOT)/out/cxl-$$name ../sim/configs/cxl_tier.py \
	    --backend-channels 8 --num-gen 1 --bus-ghz 4.0 --max-outstanding 1 \
	    $$@ >/dev/null 2>&1; \
	done
	@$(PY) -c "import re,sys; \
	  g=lambda n,k: [float(l.split()[1]) for l in open('$(ROOT)/out/cxl-%s/stats.txt'%n) if k in l]; \
	  lat=lambda n: g(n,'simSeconds')[0]/sum(g(n,'numReads::total'))*1e9; \
	  a=lat('asic')-lat('local'); f=lat('fpga')-lat('local'); \
	  print('\n  CXL-ASIC added latency: %6.1f ns   [silicon: 154]' % a); \
	  print('  CXL-FPGA added latency: %6.1f ns   [silicon: 245]\n' % f)"

cxl-bandwidth: $(BIN)
	$(RUN) --outdir=$(ROOT)/out/cxl-bw ../sim/configs/cxl_tier.py \
	  --backend-channels 8 --num-gen 16 --bus-ghz 4.0 --link-gbps 63
	@awk '/bwRead::total/ {s+=$$2} END {printf "\n  CXL 2.0 x16 effective: %.1f GB/s of 63 nominal (%.0f%%)\n\n", s/1e9, s/63e7}' \
	  $(ROOT)/out/cxl-bw/stats.txt

energy: $(BIN)
	$(PY) sim/measure_energy.py --config HBM3_16Gb_x64_1ch \
	                            --config HBM3e_24Gb_x64_1ch \
	                            --config DDR5_6400_4Gb_x8

clean:
	rm -rf $(ROOT)/out /tmp/check-*
