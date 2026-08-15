# hbm-cxl-memsys

A calibrated **gem5 + DRAMsim3** model of a GPU with on-package HBM and a
CXL-attached memory expander, together with a **static address map** that
places Mixture-of-Experts weights across the two tiers.

It answers, for a given model and HBM budget: *how long does memory take,
and how much energy does it cost, when some of the experts do not fit?*

Every number the model produces traces back to a published standard, a
vendor datasheet, or a measurement made by this repository against one of
them. `make check` re-runs all of those comparisons in one command.

Being managed collectively by:
→ Mitul Tandon (tandonmitul27)
→ Aryan Chaudhary (aryanchaudhary29)
→ Shreesh Shrinivas Nagral (bronco2910)

---

## The system

The **GPU is the CXL host**. HBM sits on its package; a CXL link runs from
that same GPU to a Type-3 memory expander. One hop, no CPU in the path.

```
         ┌──────────────────────────────┐
         │        GPU  (CXL host)       │
         │      H100 anchor, 80 GB      │
         └──┬────────────────────────┬──┘
   on-package│                        │ CXL 2.0 x16
   3540 GB/s │                        │ 50.7 GB/s effective, +160 ns
            ┌┴─────────────┐   ┌──────┴──────────────────────┐
            │ HBM3 x5      │   │ CXL Type-3 expander         │
            │ 16 ch x 64 b │   │  bridge 50 + proto 12/15 ns │
            │ 707.8 GB/s   │   │  8 x DDR5-6400 = 204.8 GB/s │
            │ per stack    │   │  backend > link, link binds │
            └──────────────┘   └─────────────────────────────┘
              NEAR TIER               FAR TIER
```

Both tiers are real simulated memory: HBM3/HBM3E and DDR5-6400 device models
run in DRAMsim3, behind a CXL protocol path calibrated against shipping CXL
silicon. HBM holds a **cache over** the CXL-resident experts rather than an
exclusive split; under the static mapping here, the resident set is fixed at
configuration time and a directory says which tier each expert is in.

→ Topology, address map and a worked placement example:
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

---

## Quick start

```bash
./setup.sh              # clone + patch + build gem5 and DRAMsim3  (~30-60 min)
make check-fast         # verify the whole tree                    (~2 min)
```

`setup.sh` clones gem5 and DRAMsim3 at pinned revisions, applies the patch in
`patches/`, installs the device configs from `configs/dramsim3/`, and builds.
Re-running it is safe; each step is skipped if already done.

**Requirements** — g++ ≥ 10, python3 ≥ 3.9, scons, cmake, make, git, zlib and
protobuf headers. Without root, conda covers all of them:

```bash
conda install -c conda-forge scons cmake zlib protobuf m4
```

---

## Running simulations

`make help` lists every entry point. Each one sets the working directory,
`LD_LIBRARY_PATH` and harness flags correctly; prefer them over hand-rolled
invocations.

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

→ Driving the gem5 harnesses directly, reading results, and the flags that
change what you measure: **[docs/USAGE.md](docs/USAGE.md)**

> **Two constraints.** Read energy through `sim/measure_energy.py` — DRAMsim3's
> raw stats are V·mA·cycles, not picojoules. And keep crossbars out of the path
> to an HBM stack (`--direct` or `--pairs`); gem5's XBar `width` is per port.
> [Details](docs/USAGE.md#two-constraints-to-respect).

---

## Documentation

| Document | What is in it |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | topology, where the weights go, a worked example |
| [docs/USAGE.md](docs/USAGE.md) | every entry point, harness flags, reading results |
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | every number, with source and confidence |
| [docs/VALIDATION.md](docs/VALIDATION.md) | what `make check` compares, and against what |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | why the device models and harness settings hold their values |

Key references behind the model: JESD238A (JEDEC HBM3) for geometry, voltages
and per-pseudo-channel timing scope; CXL-DMSim for protocol latency measured
on real CXL ASIC and FPGA silicon; O'Connor et al., MICRO'17 for the HBM
energy anchor; FloE for expert-fetch time; NVIDIA H100/H200/B200 datasheets
for the GPU anchors. Full table in
[docs/PARAMETERS.md](docs/PARAMETERS.md).

---

## Repository layout

| Path | What it is |
|---|---|
| `setup.sh` | clones, patches and builds gem5 + DRAMsim3 |
| `patches/` | our changes to gem5, as patches |
| `configs/dramsim3/` | HBM3, HBM3E and DDR5-6400 device models |
| `sim/configs/smoke_hbm.py` | near-tier harness (bandwidth, latency, energy runs) |
| `sim/configs/cxl_tier.py` | far-tier harness (link + protocol + expander media) |
| `sim/measure_energy.py` | pJ/bit measurement, with the tCK unit correction |
| `sim/check.py` | the validation suite |
| `sim/system_params.py` | every parameter, with source and confidence |
| `mapping/address_map.py` | static HBM/CXL placement |
| `mapping/geometry/` | model shapes |
| `docs/` | architecture, usage, parameters, validation, calibration |

Everything authored here is marked `@tandonmitul27` in a header comment
explaining what the file does and why it is needed.
