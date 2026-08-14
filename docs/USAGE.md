# Running simulations

<!-- @tandonmitul27 -- authored document. -->

Every entry point, what each one measures, and how to drive the gem5
harnesses directly when the make targets are not enough.

---

## The make targets

`make help` lists every entry point. Each one sets the working directory,
`LD_LIBRARY_PATH` and harness flags correctly. Prefer them over hand-rolled
invocations: a wrong flag or search path yields a plausible number rather than
an error.

| Command | What it measures |
|---|---|
| `make bw-channel` | HBM3 single-channel sequential bandwidth |
| `make bw-stack` | HBM3 full stack, 16 independent channels |
| `make cxl-latency` | CXL added latency, ASIC and FPGA devices |
| `make cxl-bandwidth` | effective bandwidth of the CXL 2.0 x16 link |
| `make energy` | pJ/bit for HBM3, HBM3E and the DDR5 media |
| `make params` | every parameter with its source and confidence |
| `make addrmap` | the static HBM/CXL map (`MODEL=`, `HBM=`) |
| `make check` | all of the above, compared against references |

`make check-fast` is the same suite with the long-running bandwidth sweeps
skipped — about two minutes instead of twenty. Use it after any edit; use the
full `make check` before trusting a result. What each check compares against
is listed in [VALIDATION.md](VALIDATION.md).

---

## Driving the harnesses directly

Both gem5 harnesses are ordinary gem5 config scripts. Run them **from the
gem5 root** — DRAMsim3 config paths inside them are relative to it:

```bash
cd gem5
export LD_LIBRARY_PATH=$PWD/ext/dramsim3/DRAMsim3:$LD_LIBRARY_PATH

# near tier ------------------------------------------------------------
build/NULL/gem5.opt ../sim/configs/smoke_hbm.py \
    --config HBM3_16Gb_x64_1ch --direct --window-ns 20000

build/NULL/gem5.opt ../sim/configs/smoke_hbm.py \
    --config HBM3_16Gb_x64_1ch --pairs 16 --sys-ghz 8.0    # full stack

build/NULL/gem5.opt ../sim/configs/smoke_hbm.py \
    --config HBM3_16Gb_x64_1ch --direct \
    --max-outstanding 1 --random                           # row-miss latency

# far tier -------------------------------------------------------------
build/NULL/gem5.opt ../sim/configs/cxl_tier.py --link-gbps 63
build/NULL/gem5.opt ../sim/configs/cxl_tier.py --link-gbps 121 --fifo 128
build/NULL/gem5.opt ../sim/configs/cxl_tier.py --no-link    # control
```

### Reading the results

Bandwidth comes out of `stats.txt` as `bwRead::total` summed across
controllers; latency is `simSeconds / numReads::total` with
`--max-outstanding 1`.

Energy is read by `sim/measure_energy.py`, **not** from DRAMsim3's stats
directly — the raw stats are in V·mA·cycles and need a tCK conversion the
simulator does not apply. See [units](CALIBRATION.md#units).

---

## Flags worth knowing

| Flag | Why it matters |
|---|---|
| `--direct` / `--pairs N` | A single crossbar in front of a stack costs ~16× the bandwidth: gem5's XBar `width` is **per port**, and one shared arbitration layer serialises every channel. `--pairs` builds independent generator↔controller pairs. |
| `--num-gen N` | One traffic generator is packet-rate limited near 128 GB/s regardless of the device. Real hosts have many outstanding requests. |
| `--fifo` | Device-side buffering. CXL 3.0 rates require 128 entries; the silicon-validated 48 cap the link at ~71 % — see [Links](PARAMETERS.md#links). |
| `--no-link` | The control for CXL runs: identical topology, link uncapped, protocol zeroed. Deliberately not a bypass — sharing the topology keeps the harness's own limits equal on both paths so they cancel in the ratio. |

---

## Two constraints to respect

Neither raises an error when violated; both yield a number rather than a
failure.

**Read energy through `sim/measure_energy.py`.** DRAMsim3's raw energy stats
are V·mA·cycles, not picojoules — a 3.2× difference at HBM3's tCK.
`measure_energy.py` applies the conversion.
→ [CALIBRATION.md](CALIBRATION.md#units)

**Do not put a crossbar in front of an HBM stack.** gem5's XBar `width` is per
port, so one shared layer serialises every channel and costs roughly 16× the
bandwidth. Use `--direct` or `--pairs`.
→ [CALIBRATION.md](CALIBRATION.md#fabric-crossbar-width-is-per-port)

[CALIBRATION.md](CALIBRATION.md) covers the rest, and is worth reading before
changing a config.
