# hbm-cxl-memsys

Two models of one system: a GPU with on-package HBM and a CXL-attached memory
expander, running Mixture-of-Experts inference when the experts do not all fit
in HBM.

| | what it is | cost per point |
|---|---|---|
| **1. Simulator** (`sim/`) | gem5 + DRAMsim3, cycle-level: HBM3/HBM3E stacks, the CXL link and protocol, the DDR5 expander | ~50 min for one decode step |
| **2. Analytical model** (`analytical/`) | closed-form time and energy for the same system, same routing logs, same placement | ~20 ms for one configuration |

They agree to ±2.4% over 103 operating points. Use the analytical model to get
results; use the simulator to measure the constants it needs and to check it.

Every number traces back to a published standard, a vendor datasheet, or a
measurement this repository makes against one of them. `make check` re-runs all
of those comparisons.

Being managed collectively by:  
→ Mitul Tandon (tandonmitul27)<br> 
→ Aryan Chaudhary (aryanchaudhary29)<br>
→ Shreesh Shrinivas Nagral (bronco2910)

---

## The system

The GPU is the CXL host. HBM sits on its package; a CXL link runs from the same
GPU to a Type-3 memory expander. One hop, no CPU in the path.

```
         ┌──────────────────────────────┐
         │        GPU  (CXL host)       │
         │      H100 anchor, 80 GB      │
         └──┬────────────────────────┬──┘
   on-package│                        │ CXL 2.0 x16
   3683 GB/s │                        │ 50.7 GB/s effective, +160 ns
            ┌┴─────────────┐   ┌──────┴──────────────────────┐
            │ HBM3 x5      │   │ CXL Type-3 expander         │
            │ 16 ch x 64 b │   │  bridge 50 + proto 12/15 ns │
            │ 736.6 GB/s   │   │  8 x DDR5-6400 = 204.8 GB/s │
            │ per stack    │   │  backend > link, link binds │
            └──────────────┘   └─────────────────────────────┘
              NEAR TIER               FAR TIER
```

HBM holds a cache over the CXL-resident experts, not an exclusive split. Under
the static mapping here the resident set is fixed at configuration time.

→ Topology, address map, worked example: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## How the two models relate

```
  synthetic traffic  ──▶  sim/  ──▶  bandwidth, latency, pJ/bit
                                          │
                                          ▼
                                    analytical/  ──▶  results
                                          │
                       a schedule of barriers
                                          ▼
                              sim/configs/moe_replay.py
                                          │
                                          ▼
                            validation/  ──▶  103 points, ±2.4%
```

The simulator is used two ways. Synthetic traffic measures the hardware and
gives the constants the analytical model uses. A real MoE schedule — emitted by
the analytical model — is replayed barrier by barrier to check it.

This is why both live in one repository: `moe_replay.py` cannot run without a
schedule from `analytical/trace_gen.py`, and both read the same placement from
`mapping/address_map.py`.

Simulating everything is not an option. One replay point takes ~50 minutes, the
longest took 13 hours, and the full campaign was 226 core-hours. The sweep is
80,640 configurations.

→ The analytical model: **[docs/MODEL.md](docs/MODEL.md)**

---

## Quick start

```bash
# Model 2 only -- no build needed
make model MODEL=Mixtral-8x7B HBM=80     # one operating point
make sweep                               # the full grid, ~30 min

# Model 1 as well -- needed to re-measure or re-validate
./setup.sh              # clone + patch + build gem5 and DRAMsim3  (~30-60 min)
make check-fast         # verify the whole tree                    (~2 min)
```

`setup.sh` clones gem5 and DRAMsim3 at pinned revisions, applies the patches in
`patches/`, installs the device configs from `configs/dramsim3/`, and builds.
Re-running it is safe.

**Requirements** — g++ ≥ 10, python3 ≥ 3.9, scons, cmake, make, git, zlib and
protobuf headers. Without root, conda covers all of them:

```bash
conda install -c conda-forge scons cmake zlib protobuf m4
```

---

## Running them

`make help` lists every entry point. Each sets the working directory,
`LD_LIBRARY_PATH` and harness flags correctly, so prefer them over running the
scripts by hand.

### Model 2 — the analytical model

Needs no gem5. Skip `./setup.sh` if you only want results.

| Command | What it does |
|---|---|
| `make model` | one operating point (`MODEL=`, `HBM=`, `BATCH=`, `POLICY=`, `PHASE=`, `GPU=`) |
| `make sweep` | the full 80,640-point grid → `results/sweep.parquet` (~30 min) |
| `make confidence` | accuracy, parameter sensitivity and coverage of the model |
| `make routing` | integrity of the routing logs the model reads |

### Model 1 — the simulator

Needs a built gem5. Run these when the hardware changes, or to re-validate
after changing the analytical model.

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

Replaying a real MoE workload is not a `make` target: it needs a schedule and
hours of wall time. See `validation/README.md`.

→ Running both models, every flag, and the pitfalls: **[docs/USAGE.md](docs/USAGE.md)**

<!-- 
> **Two traps.** Read energy through `sim/measure_energy.py` — DRAMsim3's raw
> stats are V·mA·cycles, not picojoules. And keep crossbars out of the path to
> an HBM stack (`--direct` or `--pairs`); gem5's XBar `width` is per port.
> Neither raises an error; both give a wrong number.
> [Details](docs/USAGE.md#two-constraints-to-respect). -->

---

## Documentation

**Start here**

| Document | What is in it |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | **how to run both models**: entry points, flags, pitfalls |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the system: topology, where the weights go, a worked example |

**Model 1 — the simulator**

| Document | What is in it |
|---|---|
| [sim/README.md](sim/README.md) | harness mechanics, the three gem5 patches, the traps |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | why the device models and harness settings hold their values |
| [docs/VALIDATION.md](docs/VALIDATION.md) | what `make check` compares, and against what |

**Model 2 — the analytical model**

| Document | What is in it |
|---|---|
| [docs/MODEL.md](docs/MODEL.md) | what is modelled, where the numbers come from, what it cannot claim |
| [results/README.md](results/README.md) | the sweep, its headline numbers, and how to read them |

**Shared, and how the two compare**

| Document | What is in it |
|---|---|
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | every number, with source and confidence |
| [docs/VALIDATION_CAMPAIGN.md](docs/VALIDATION_CAMPAIGN.md) | the 103-point campaign: confidence, coverage, known residuals |
| [validation/README.md](validation/README.md) | the campaign drivers, and what each produced |
| [data/README.md](data/README.md) | the routing logs: schema, provenance, traps in reading them |
| [docs/analytical_model_spec.md](docs/analytical_model_spec.md) | ⚠️ **stale** — an earlier design note; its constants predate the campaign |

Main references: JESD238A (JEDEC HBM3) for geometry, voltages and timing scope;
CXL-DMSim for protocol latency measured on real CXL ASIC and FPGA silicon;
O'Connor et al., MICRO'17 for the HBM energy anchor; FloE for expert-fetch
time; NVIDIA H100/H200/B200 datasheets for the GPU anchors. Full table in
[docs/PARAMETERS.md](docs/PARAMETERS.md).

---

## Repository layout

Grouped by which model each path belongs to.

**Model 1 — the simulator**

| Path | What it is |
|---|---|
| `setup.sh` | clones, patches and builds gem5 + DRAMsim3 |
| `patches/` | our three changes to gem5, as patches |
| `configs/dramsim3/` | HBM3, HBM3E and DDR5-6400 device models |
| `sim/configs/smoke_hbm.py` | near-tier harness (bandwidth, latency, energy runs) |
| `sim/configs/cxl_tier.py` | far-tier harness (link + protocol + expander media) |
| `sim/configs/moe_replay.py` | replays a real MoE schedule on gem5, barrier by barrier |
| `sim/measure_energy.py` | pJ/bit measurement, with the tCK unit correction |
| `sim/check.py` | the validation suite (22 checks) |
| `sim/system_params.py` | every parameter, with source and confidence |
| `sim/README.md` | harness mechanics, the gem5 patches, the traps |

**Model 2 — the analytical model**

| Path | What it is |
|---|---|
| `analytical/trace_gen.py` | the timing model: routing log + placement + policy → time |
| `analytical/gpu_model.py` | GPU roofline and per-barrier compute floor |
| `analytical/energy_model.py` | pJ/bit and state-resolved background power → energy, EDP |
| `analytical/sweep.py` | the configuration grid |
| `docs/MODEL.md` | what is modelled, where the numbers come from, what it cannot claim |

**Shared by both**

| Path | What it is |
|---|---|
| `mapping/address_map.py` | static HBM/CXL placement; one copy, read by both models |
| `mapping/geometry/` | model shapes |
| `data/routing/` | the workload: 20 routing logs (43 MB), plus an integrity check |

**Where they are compared**

| Path | What it is |
|---|---|
| `validation/` | the replay campaign drivers |
| `docs/replay_points.csv` | 103 points: gem5 measurement vs analytical prediction |
| `docs/confidence_analysis.py` | regenerates the accuracy and coverage numbers from that file |
| `docs/VALIDATION_CAMPAIGN.md` | confidence, coverage, and known residuals |
| `results/` | the headline tables; the parquet itself is generated by `make sweep` |

Files authored here carry a header comment saying what they do and why.
