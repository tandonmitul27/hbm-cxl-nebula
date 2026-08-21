# Calibration notes

<!-- @tandonmitul27 -- authored document. -->

Why the device models and harness settings hold the values they do, and which
of those values are load-bearing. Read this before changing a config.

---

## Why the defaults are not used

Stock gem5 and DRAMsim3 settings report roughly **1/16** of an HBM3 stack's
achievable bandwidth. Three settings are responsible, and none of them raise
an error when wrong:

1. fabric — a `SystemXBar` at default width in front of the memory
2. address mapping — `rorabgbachco`, which lands sequential traffic on one bank group
3. queue depth — `trans_queue_size = 32`, which leaves the device latency-bound

Each is covered below. Because a misconfiguration here produces a number
rather than a failure, any new configuration is treated as wrong until
compared against something external — which is what `make check` is for.
See [VALIDATION.md](VALIDATION.md).

---

## The near tier

### Fabric: crossbar width is per port

gem5's `XBar` `width` parameter is **per port**, not shared. A crossbar with
many requestor ports therefore models a switch, not a bus — and its single
arbitration layer serialises every channel behind it. The visible effect is
that adding channels stops helping.

`--direct` removes the fabric entirely; `--pairs N` builds N independent
generator↔controller pairs. Per-channel bandwidth then scales **exactly**
linearly: 46.0 GB/s × 16 = 736.6 GB/s.

Related: several generators over one address space march in lockstep and
hammer the same channel simultaneously, which is indistinguishable from poor
memory scaling. Every generator is therefore staggered by one interleave
stride.

### Address mapping

`rorabgbacoch` (row, rank, bank group, bank, column, channel) puts
consecutive cache lines on different channels and keeps a row open across a
burst. The stock `rorabgbachco` costs roughly 40 % of achievable bandwidth on
sequential traffic.

Corroboration: Samsung's DVCon India 2023 HBM3 verification paper reports the
same dependence of achieved bandwidth on bank-group interleaving and
transaction size.

### Queue depth

`trans_queue_size = 128` is the knee of a measured sweep. Below it the device
is request-limited; above it nothing improves.

This interacts with a wrapper limitation: gem5's DRAMsim3 wrapper caps *total*
outstanding requests at the scalar `trans_queue_size`, so **one 16-channel
instance is request-limited long before the device is**. A full stack must be
built from 16 independent single-channel instances, which is why
`HBM3_16Gb_x64_1ch.ini` exists.

### Timings

Converted from Ramulator2's JESD238-cited HBM3 preset at tCK = 0.3125 ns. The
preset is the source of record for array timings; JESD238A fixes their scope
but not their values.

`tFAW` and `tRRD` are **halved** relative to their channel-level values,
because JESD238A Table 3 / §3.1.2 scopes array timings *per pseudo-channel*
while we model at channel level.

`columns = 64`, not 32: DRAMsim3 internally doubles the column count for HBM's
prefetch, and 32 trips a capacity assertion.

Note on clock domains: our tCK is the data-strobe domain, JEDEC's is the
command clock (0.625 ns at 6.4 Gbps). Cycle counts therefore differ from the
standard's tables while every nanosecond value matches.

### Validation

736.6 GB/s per stack, against 670 GB/s per stack implied by the H100 SXM5
datasheet (3350 GB/s ÷ 5 stacks). Shipping parts clock HBM below the top JEDEC
bin, so measuring slightly above the datasheet figure at the top bin is the
expected direction.

---

## The far tier

### Reproducing a published model instead of forking

The published CXL device model (CXL-DMSim / SimCXL) is a **fork of gem5 23.1**,
so it cannot be combined with 25.1. Rather than freeze on an old gem5, the
model is reproduced from its published parameters on stock SimObjects:

| Component | Parameter | Value |
|---|---|---|
| host bridge | `bridge_lat` | 50 ns |
| host bridge | `proto_proc_lat` | 12 ns |
| device | `proto_proc_lat` | 15 ns (ASIC), 60 ns (FPGA) |
| device | req/rsp FIFO | 48 entries |

gem5's `Bridge` pays its `delay` in each direction, so 2 × (50 + 12 + 15) =
**154 ns** round trip by construction.

### Three requirements the far-tier harness must meet

**The link must be a single-port stage.** Per-port `width` semantics apply
here as in the near tier: without a single-port stage, link rate has no effect
on measured bandwidth. Every requestor is aggregated through a wide bus and
then crosses one stage whose single layer serialises all traffic. Only then is
bandwidth `width × clock`.

**Backend channels must be interleaved.** With disjoint ranges a linear sweep
occupies one channel at a time, which is indistinguishable from a bandwidth
ceiling.

**The baseline must share the CXL path's topology.** `--no-link` builds the
identical topology with the link uncapped and protocol latency zeroed, rather
than bypassing the bridges. Bypassing would give the baseline *less*
outstanding capacity than the CXL path, which gains FIFO entries per bridge.
Sharing the topology makes the harness's own limits affect both paths equally,
so they cancel in the ratio.

### CXL 3.0 needs deeper buffering

Effective bandwidth through the full path, measured at exact-nominal link
stages: PCIe4 x16 25.3 GB/s (97 % of 26), CXL2 x16 50.7 (80 % of 63).

Device buffering sets a Little's-law ceiling of its own:

```
eff_link = min(link rate, 8 bridges x FIFO entries x 64 B / RTT)
```

At CXL3 x16 the silicon-validated 48 entries bind:

```
outstanding needed = bandwidth × round-trip ÷ line size
                   = 121 GB/s × ~280 ns ÷ 64 B  ≈  530 requests
```

far beyond 8 bridges × 48. With 128-entry FIFOs the link reaches 106.6 GB/s
(88 %), matching the lower rates' efficiency. No shipping CXL 3.0 expander
exists to validate the deeper figure against, so any CXL 3.0 result must state
the buffering assumption.

Note the cap depends on the traffic pattern. The ~86 GB/s (71 %) figure often
quoted for 48-entry FIFOs is a **sustained-mix** measurement; large sequential
expert fetches on the same buffers reach ~101 GB/s (84 %), because a shallower
window queues less and turns over faster (RTT ~242 ns rather than the
deep-FIFO ~280 ns). State which pattern a CXL3 number refers to.

Two related harness notes: the link crossbar's whole-cycle packet occupancy
quantises bandwidth, so fast links need a fast `--bus-ghz` for the quantum to
stay fine relative to the link rate; and the DDR5 config uses the same
`trans_queue_size = 128` knee as the HBM calibration.

### Validation

| | Ours | Silicon |
|---|---|---|
| CXL ASIC added latency | 159.7 ns | 154 ns |
| CXL FPGA added latency | 248.5 ns | 245 ns |
| Mixtral expert over PCIe4 x16 | 15.0 ms | ~15 ms (FloE) |

---

## Energy

DRAMsim3 computes energy the standard way — IDD × VDD × time-in-state, the
Micron power-calculation methodology — and writes per-channel totals to
`dramsim3.json` when the wrapper is destroyed, **into the DRAMsim3 directory,
not the gem5 output directory**.

### Units

DRAMsim3's energy stats are **V·mA·cycles, not picojoules**. Its energy
increments carry timing in clock cycles and nothing in the energy path
multiplies by tCK (only `average_power` is unit-correct, because the cycles
cancel there).

```
true pJ = reported × tCK_ns
```

At the stock DDR4 tCK = 0.63 ns the discrepancy is small enough to overlook;
at HBM3's 0.3125 ns it is 3.2×. `sim/measure_energy.py` applies the
correction, and is the only supported way to read energy out of this repo.

### Calibrated-to-published HBM currents

JESD238A fixes VDD (Table 70: 1.1 V) and defines the IDD states (§9), but
leaves the current *values* to vendor datasheets — Table 90 is an empty
template. So the HBM `[power]` sections keep the HBM2 template's state
*ratios* and scale the whole current family by one factor:

| | Factor | Result |
|---|---|---|
| HBM3 | k = 3.270 | 3.97 pJ/bit |
| HBM3E | k = 4.648 | 3.97 pJ/bit |

against the published anchor of **3.97 pJ/bit** (O'Connor et al., MICRO'17).
One target, one knob, stated as such. HBM3E is held equal per bit — no vendor
publishes HBM3E pJ/bit, and Micron claims better perf/W, so equal is the
conservative choice — and consequently **no generation-relative HBM energy
claims are made**.

Stack-level sanity: 1.40 W/channel × 16 ≈ 22 W/stack at full bandwidth →
~112 W for five stacks, consistent with 3.9 pJ/bit × 3.35 TB/s = 104 W.

### The DDR5 media, uncalibrated

`DDR5_6400_4Gb_x8.ini` was authored from gem5 25.1's `DDR5_6400_4x8` class,
whose timings come from the Micron DDR5 core datasheet and whose IDD currents
come from the Micron 16 Gb DDR5 die-rev-A datasheet at the DDR5-4800 grade
(gem5's own convention — the fastest grade with published IDD).

DRAMsim3 has no DDR5 protocol enum, so DDR5 is expressed as `protocol = DDR4`
(bank groups, same refresh semantics) with DDR5 geometry, BL16 and DDR5
timings.

Nothing was tuned. Measured **11.8 pJ/bit** sequential, inside the DDR/GDDR
device-class range — a check rather than a fit.

Two documented understatements: the DRAMPower-style activate term goes slightly
negative with these currents (IDD0 is measured cycling one bank while IDD3N
assumes all-bank active standby) and is clamped to zero; and the gem5 class
carries no IPP/VPP current.

### What DRAMsim3 cannot model

The CXL SerDes and controller. No DRAM simulator models a serial link, and no
vendor publishes the split between PHY and controller. `system_params.py`
brackets it:

| Bound | Value | Source |
|---|---|---|
| short-reach PHY | ~0.5 pJ/bit/direction | on-package link literature |
| long-reach PCIe 5.0 PHY | 11.4 pJ/bit | CICC 2020 |
| whole-device ceiling | 18.75 pJ/bit | 30 W ÷ 200 GB/s, Structera X brief |

Sweep it over 2–19 pJ/bit and report conclusions across the range.

### Measurement mechanics

Energy runs are single-instance (`--direct`) because multiple DRAMsim3
instances collide on the single `dramsim3.json`. Per-channel energy is
identical under symmetric interleave, so totals scale by channel count.

---

## What remains an estimate

Listed in `system_params.py` under `NEEDS_SOURCE` so they cannot reach a result
unnoticed:

- **HBM3E array timings** — held from HBM3 in nanoseconds with only the PHY
  scaled. Same array generation, defensible, not sourced.
- **GPU CXL lane count** — x16 assumed; no shipping GPU exposes CXL host ports.
  Sweep it.
- **CXL link/controller energy split** — bracketed above, never fixed.
