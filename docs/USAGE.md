# Running the two models

<!-- @tandonmitul27 -- authored document. -->

This repository holds two models of the same system. **Part 1** is the gem5 +
DRAMsim3 simulator; **Part 2** is the analytical model. If you only want
answers, skip to Part 2 — it needs no gem5 and no build.

---

# Part 1 — the simulator

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
| `--fifo` | Device-side buffering. CXL 3.0 rates require 128 entries; the silicon-validated 48 cap **fetch** traffic at ~101 GB/s (84 %) — a sustained-traffic measurement previously quoted ~86 GB/s (71 %), which does not hold for the bursty per-barrier fetches the workload actually issues. See [Links](PARAMETERS.md#links). |
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

---

## Replaying a real MoE workload

The harnesses above drive *synthetic* traffic. `sim/configs/moe_replay.py`
drives the real thing: a schedule of per-barrier byte counts produced by the
analytical model, replayed barrier by barrier with the HBM stream and the CXL
fetch racing each other exactly as the recurrence says they should.

```bash
# 1. the analytical model emits the schedule
python analytical/trace_gen.py --tag OLMoE-1B-7B --batch 1 --hbm-gib 1.53 \
    --policy static --phase decode --dtype fp16 --max-barriers 8 \
    --t0-us 0 --emit-schedule /tmp/sched.json

# 2. gem5 replays it
cd gem5
export LD_LIBRARY_PATH=$PWD/ext/dramsim3/DRAMsim3:$LD_LIBRARY_PATH
build/NULL/gem5.opt ../sim/configs/moe_replay.py \
    --schedule /tmp/sched.json --hbm-config HBM3_16Gb_x64_1ch
# REPLAY barriers=8 total=4.272 ms  analytic=4.306 ms  ratio=0.992
```

The printed `ratio` is gem5 divided by the analytical prediction — the single
number the whole validation campaign is made of.

| Flag | Why it matters |
|---|---|
| `--hbm-channels N` | how many of the stack's 16 channels to instantiate. The per-channel byte share is always that of the real stack, so this trades simulation time against event count, **not** accuracy. Default 4. |
| `--fifo`, `--bus-ghz` | must match the link the schedule was emitted for: 48/4.0 for CXL 2.0, 128/15.125 for CXL 3.0, 48/6.5 for PCIe 4.0. Mismatch silently measures a different link. |
| `--energy-dir DIR` | give each DRAMsim3 instance its own output directory so HBM and expander energy can be summed. Without it they overwrite one file. |
| `--fill-writes` | also issue the cache-fill write of each missed expert into HBM. Off by default so the read-only timing campaign stays reproducible. |

**Two things to know before reading a replay number.**

*Ratios are conventionally t0-free* (`--t0-us 0`). `t0` is exact arithmetic on
both sides, but it is serial in the model while the replay's gate is
`max(stream, floor)` — leaving it in measures bookkeeping, not the memory
recurrence.

*The replay quantum is a real artifact.* `ratio = 0.9556 + 0.666 x
(quantum / t_barrier)`. Changing only `--quantum-ns` moved one ratio from
1.023 to 1.228. Keep it at the default unless you are deliberately measuring
the artifact, and never tune the model against it.

Wall time is the binding constraint: a median point is ~50 minutes and the
largest in the campaign took 13 hours. `validation/` holds the drivers that
ran the 103 points.

---

# Part 2 — the analytical model

Needs no gem5. Reads a routing log, applies a placement policy, and returns
time and energy for the whole run.

```bash
make model MODEL=Mixtral-8x7B HBM=80          # the H100 anchor
python analytical/trace_gen.py --tag Mixtral-8x7B --batch 16 --hbm-gib 80 \
    --policy static --phase decode --dtype fp16 --gpu H100
```

```
  bytes fetched        234.28 GiB
  compute time         822.29 ms
  stall time          5030.16 ms  (92.6% of total)
  total time          5433.72 ms
  per decode step     169.804 ms  (ideal 25.696 ms)
```

**`total time` is what you quote.** `compute time` is the stall-free floor —
what the run would cost if every expert were resident. They are equal only
when nothing misses. An early draft of these results quoted the floor as if
it were achieved time and understated Mixtral by 6.6x.

## The axes

| Flag | Meaning |
|---|---|
| `--tag` | which model: `OLMoE-1B-7B`, `DeepSeek-V2-Lite`, `Phi-3.5-MoE`, `Mixtral-8x7B`, or the dense control `Qwen2.5-3B-dense` |
| `--batch` | 1, 4, 16 or 64 — must be a batch the routing log was recorded at |
| `--hbm-gib` | HBM capacity in GiB, **absolute**. This is the main experimental axis |
| `--gpu` | `H100` / `H200` / `B200`; sets datasheet FLOPS and stack count |
| `--dtype` | `fp16` or `fp8`; fp8 halves every weight and doubles peak FLOPS |
| `--phase` | `decode` or `prefill`. Prefill unions experts across the whole prompt — the harshest tiering case |
| `--policy` | `static` (fixed resident set), `lru`, `lru_layer`, `oracle`, or `none` (nothing resident) |
| `--prefetch-depth` | layers of lookahead. **Staged experts occupy cache slots**, so depth reduces effective capacity by `depth x top_k` |
| `--link-gbps` | 24.9 (PCIe 4.0 x16), 50.7 (CXL 2.0 x16), 114.5 (CXL 3.0 x16) |

Everything else — `--hbm-gbps`, `--stacks`, `--t0-us`, `--xfer-lat-ns`,
`--fifo-entries` — defaults to the calibrated value for the chosen GPU and
link. Change them only to sweep a parameter deliberately;
`python sim/system_params.py` prints every default with its provenance.

## Sweeping

```bash
make sweep        # 80,640 rows -> results/sweep.parquet, ~30 min
```

One row per configuration, with energy and EDP at `e_link` = 2, 10 and 19
pJ/bit (columns `energy_j_e2`, `edp_js_e10`, …). The columns that matter:

| Column | Meaning |
|---|---|
| `total_s` | achieved time — **the answer** |
| `compute_s` | the stall-free floor |
| `stall_frac` | fraction of `total_s` spent waiting on the link |
| `hit_rate` | fraction of expert demands served from HBM |
| `layers` | barriers in the window; `total_s / (layers / moe_layers)` is ms per decode step |
| `bytes_fetched` | bytes pulled over CXL |
| `cap_kind` | `abs` = the GPU's native HBM, `frac` = a fraction of the model's expert working set |

`results/README.md` has the headline tables and how to read them without
tripping over the traps.

## Pitfalls

**Residency is not hit rate.** A model that fits entirely means `static` *is*
the compute floor, and a dynamic policy can only lose ground by evicting
something it needed — hence `lru_layer` running 1.5–3.6x worse at full
residency. That is a property of the policy, not of the tier.

**Quote energy across the `e_link` bracket.** The CXL SerDes + controller cost
is not public and is swept 2–19 pJ/bit. It moves energy by up to 1.40x end to
end. Orderings are preserved; absolute Joules are not a single number.

**Check the coverage before trusting a new point.** The ±2.4% accuracy claim
holds inside the validated envelope — the models, links, batches, dtypes,
phases, policies and GPUs listed in
[VALIDATION_CAMPAIGN.md](VALIDATION_CAMPAIGN.md) §C. Outside it you are
extrapolating, and should say so.

**HBM3e systems carry a +1.5% bias.** Measured, documented, unexplained; see
[MODEL.md](MODEL.md) §5b.
