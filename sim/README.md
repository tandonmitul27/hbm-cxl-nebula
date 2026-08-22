# Simulator setup and calibration

gem5 25.1.0.1 (NULL ISA) + DRAMsim3, trace-driven, no CPU. SimCXL cloned for
its CXL device model and parameters.

## System modelled

**The GPU is the CXL host.** It has HBM on package and a CXL link to a memory
expander — one hop.

```
    GPU  ─┬─ HBM             on package, 16 channels    708 GB/s/stack
          │
          └─ CXL 2.0 x16 ──  memory expander            52 GB/s
```

This differs from shipping hardware, where CXL attaches to the **CPU** and
reaching the GPU costs a second PCIe traversal (`CXL → CPU DRAM → PCIe → GPU
HBM`, two narrow hops in series). GPU-side CXL is emerging rather than
deployed — a 2026 kernel patch adds a CXL DVSEC readiness check for
Blackwell-Next in NVIDIA's NVGrace-GPU VFIO driver — so **D1 must state which
is assumed.** Ours is the forward-looking single-hop case, which is the
*optimistic* one for CXL: it omits a hop that would otherwise halve effective
bandwidth and add latency.

The two tiers are measured separately and combined analytically in Phase 2:

| tier | harness | result |
|---|---|---|
| HBM | `smoke_hbm.py --pairs 16` | 708 GB/s/stack |
| CXL | `cxl_tier.py --link-gbps 63` | 52 GB/s |

`cxl_tier.py --no-link` is a **control**, not a local-memory baseline: it is
the expander's own DRAM with the link and protocol removed, used to isolate
what the CXL path costs. Local memory in this system is HBM.

All hardware parameters, with provenance and confidence, are in
[`system_params.py`](system_params.py); run it to audit them. Three are still
flagged `ESTIMATE` and must not reach the report unverified.

## Build

```bash
source sim/build-env.sh          # conda-provided zlib/protobuf/m4; no sudo here
cd sim/gem5 && scons build/NULL/gem5.opt -j16 --ignore-style
```

Two things were needed to get here:

- **DRAMsim3 must live at `gem5/ext/dramsim3/DRAMsim3/`** and be built with
  cmake first; gem5's SConscript links `libdramsim3.so` from that directory.
- **`src/mem/dramsim3_wrapper.hh` needed `#include <cstdint>`.** GCC 13 no
  longer includes it transitively, so `uint64_t` was undeclared. One line.

No sudo on this box, so zlib, protobuf and m4 come from conda via `CPATH` /
`LIBRARY_PATH` (see `build-env.sh`).

## Calibration — read this before trusting any bandwidth number

A saturating sequential read against `HBM2_8Gb_x128` (8 channels, 128-bit,
tCK 1 ns → **256 GB/s theoretical**):

| configuration | achieved | note |
|---|---|---|
| stock SystemXBar, stock config | 12.1 GB/s | 5% of peak |
| direct connect (no crossbar) | 43.5 GB/s | crossbar was the cap |
| + channel-interleaved mapping | 88.0 GB/s | `rorabgbacoch` |
| + deeper queues (128/32) | **200.0 GB/s** | 78% of peak |

**Out of the box, this stack understates HBM bandwidth by 16x.** Three
independent throttles, none of them DRAM physics:

1. **`SystemXBar` has `width = 16` bytes** at 1 GHz, so a 64 B line takes 4
   cycles to serialize — a hard 16 GB/s ceiling regardless of the memory behind
   it. There are no caches in a trace-driven setup, so coherence is
   unnecessary: use a wide `NoncoherentXBar`, or connect directly when there is
   only one memory.
2. **Address mapping `rorabgbachco` places channel bits above column**, so a
   sequential stream stays in one channel for 4 KB at a time instead of
   spreading across all eight. An expert fetch is *exactly* a large sequential
   burst, so this matters more here than for typical workloads. Interleaving
   channels low (`rorabgbacoch`) doubles throughput.
3. **Transaction/command queues (32/8) bound outstanding requests**, and by
   Little's law bound bandwidth. Deepening them to 128/32 is what closes the
   gap to a realistic 78% streaming efficiency.

This matters beyond tidiness: a crippled HBM tier makes CXL look *competitive*,
so the error flatters the result. Any HBM-vs-CXL ratio computed on the stock
configuration would have been wrong by more than an order of magnitude.

**Calibration rule adopted:** configure so achieved sequential-read bandwidth
lands at published streaming efficiency for the device (~75-85% of peak), and
report the configuration alongside every result. This mirrors what CXL-DMSim
did against real silicon.

## HBM3 / HBM3e models (authored)

DRAMsim3 ships nothing newer than HBM2, so HBM3 and HBM3e were authored:
`configs/dramsim3/HBM3_16Gb_x64.ini`, `configs/dramsim3/HBM3e_24Gb_x64.ini` (plus `_1ch`
variants used for calibration).

`protocol = HBM` is retained — DRAMsim3 has no HBM3 protocol, and the
generational differences are geometry and timing, both config-expressible:

| | HBM2 | HBM3 | HBM3e |
|---|---|---|---|
| channels x width | 8 x 128b | 16 x 64b | 16 x 64b |
| data rate | 2.0 Gbps (DRAMsim3 stock) | 6.4 Gbps | 9.6 Gbps |
| tCK | 1 ns | 0.3125 ns | 0.2083 ns |
| BL (→ 64 B access) | 4 | 8 | 8 |
| capacity/stack | 8 GB | 16 GB | 24 GB |
| theoretical | 256 GB/s | 819 GB/s | 1229 GB/s |

### Provenance: which fields are spec-derived, which are ours

Capacity is **asserted at generation time** rather than assumed — the assertion
caught a wrong `columns` value on the first pass.

**Both geometry and timings are JESD238-sourced**, taken from Ramulator2's
HBM3 model (`python/ramulator/dram/hbm3.py`), which cites JESD238 Table 4 for
organization and Table 84 for tRFC. Ramulator2 expresses timings in its own
command-clock cycles (tCK = 625 ps); they are converted to nanoseconds and then
to DRAMsim3 cycles at tCK = 0.3125 ns.

| field | JESD238 | config | source |
|---|---|---|---|
| channels per stack | 16 | 16 | spec |
| channel width | 64-bit (2 x 32b pseudo-ch) | `device_width = 64` | spec |
| banks per channel | 32 (2 pch x 4 BG x 4 banks) | `bankgroups=8, banks_per_group=4` | spec |
| row / page | 1 KB | `columns=64` -> 1024 B | spec |
| access granularity | 64 B (BL8 on 64-bit) | `BL = 8` | spec |
| data rate | 6.4 Gbps | `tCK = 0.3125 ns` | spec |
| stack capacity | 16 GB (8-hi, 16Gb) | 16 GB, **asserted** | spec |
| `tCL` | 12.5 ns | `CL = 40` | Ramulator2 preset |
| `tRCD` RD / WR | 19.375 / 9.375 ns | 62 / 30 | Ramulator2 preset |
| `tRP` | 16.25 ns | 52 | Ramulator2 preset |
| `tRAS` | 28.125 ns | 90 | Ramulator2 preset |
| `tFAW` | 15 ns | 48 | Ramulator2 preset |
| `tRRD_S` / `tRRD_L` | 2.5 / 3.125 ns | 8 / 10 | Ramulator2 preset |
| `tRFC` | 350 ns (8 Gb/channel) | 1120 | JESD238 Table 84 |
| `tREFI` | 3.9 us | 12480 | JESD238 |

This corrected several values that an earlier hand-derivation got wrong —
notably `tCL` (14 -> 12.5 ns), `tRCD` (14 -> 19.4), `tFAW` (30 -> 15) and
`tRFC` (260 -> 350).

Still ours, and to be stated as such:

- **HBM3E timings.** Ramulator2 has no HBM3E preset, so HBM3's array timings
  are held constant in nanoseconds and only the interface is scaled to
  9.6 Gbps. Defensible — same array generation, faster PHY — but not sourced.
  `tRFC` moves to 450 ns per the Table 84 density tier.
- **Pseudo-channels are approximated, not modelled.** DRAMsim3 has no
  pseudo-channel concept. Modelling 32 x 32-bit pseudo-channels would force 32 B
  granularity and hence a 32 B gem5 cacheline (DRAMsim3 asserts burst ==
  cacheline), which doubles packet count for the same bytes and loses the
  command bus shared between a pseudo-channel pair. At channel granularity the
  peak bandwidth and 64 B granularity both come out right, and all 32 banks are
  carried.

  **One correction was required.** JESD238 applies `tFAW` and `tRRD` *per
  pseudo-channel* — Ramulator2 declares them at `level="PseudoChannel"` — so a
  64-bit channel really has **two independent activation budgets**. Modelling
  the pair as one channel gave it a single budget and halved the achievable
  activation rate. Halving `tFAW` and `tRRD` at channel level restores the
  correct aggregate. Measured effect, exactly as theory predicts:

  | access pattern | one budget | two budgets | change |
  |---|---|---|---|
  | sequential (expert fetch) | 44.2 GB/s | 44.3 GB/s | +0.2% |
  | random (activation-bound) | 15.1 GB/s | 29.9 GB/s | **+98%** |

  Full-stack sequential bandwidth is unchanged (736.6 / 1072.1 GB/s), because
  streaming needs roughly one activation per 20 ns per channel while `tFAW`
  permits eight per 15 ns — it never binds. The correction is adopted anyway:
  it costs nothing and matters as soon as any scattered access is modelled
  (KV cache, fine-grained expert access, CXL-side random traffic).

  Residual inaccuracy: all 32 banks now share one doubled budget rather than
  two isolated 16-bank budgets, which is marginally optimistic for adversarial
  patterns that concentrate on a single pseudo-channel.
- **Preamble (`tRPRE`/`tWPRE` = 2) and power-down timings** (`tXS`, `tCKE`,
  `tCKSRE`, `tXP`) are not in the Ramulator2 preset; none are exercised by
  read-only streaming.
- **Queue depths** are the one remaining non-datasheet value, but they are now
  chosen by a stopping rule rather than by taste. Sweeping `trans_queue_size`
  against sequential bandwidth (single channel, HBM3):

  | `trans_queue_size` | seq GB/s | % of channel peak |
  |---|---|---|
  | 8 | 14.3 | 28% |
  | 16 | 19.3 | 38% |
  | 32 | 27.8 | 54% |
  | 64 | 41.3 | 81% |
  | **128** | **44.3** | **86%** |
  | 256 | 44.7 | 87% |

  128 is the **knee**: doubling it again buys 0.9%. The rule applied is "raise
  the queue until the *model* saturates, then stop" — not "raise it until the
  answer looks right". 256 confirms saturation.

  The depth is forced by gem5, not by the DRAM: the wrapper reuses
  `trans_queue_size` as the **total outstanding-request cap**, so by Little's
  law bandwidth is `outstanding x 64 B / loaded latency`. At 128 that is 8 KB
  against ~185 ns of loaded latency, which is what sustains 44 GB/s. As a
  DRAMsim3 `PER_BANK` depth 128/bank is generous; as a per-channel outstanding
  limit it is unremarkable. The wrapper conflates the two and the larger value
  wins.

  `cmd_queue_size` was swept too and is **irrelevant** here (8 vs 32 gives
  44.2 vs 44.3 GB/s), so it is set to the more realistic 8.

## The outstanding-request trap (why multi-channel numbers were wrong)

gem5's wrapper caps *total* outstanding requests at `wrapper.queueSize()`, and
`MemorySystem::GetQueueSize()` returns `trans_queue_size` — a **single scalar**,
not scaled by channels or banks. So a 16-channel HBM3 device gets the same
request budget as a 1-channel one, and throughput becomes latency-bound:

```
BW = queueSize x 64 B / latency  ~=  8 KB / 30 ns  ~=  265 GB/s   (a hard cap)
```

That is why full-device HBM3 (259 GB/s) and HBM3e (270 GB/s) measured almost
identically despite a 50% clock difference — a clear tell that the *device* was
not the bottleneck. Any "HBM3 gives us X" number taken from that configuration
would be wrong by ~3x.

**Per-channel, in isolation** (one generator, one channel, no crossbar):

| device | per channel | % of channel peak |
|---|---|---|
| HBM2 | 29.2 GB/s | 91% |
| HBM3 | 45.3 GB/s | 88% |
| HBM3e | 67.8 GB/s | 88% |

88-91% streaming efficiency is realistic and scales correctly with interface
rate — the check the pre-calibration numbers failed.

### Full-stack bandwidth, and the crossbar artifact behind it

Measured with **independent per-channel controllers** (`--pairs N`: N disjoint
generator/memory pairs, each on its own address range, no shared fabric):

| device | channels | per stack | % of peak | timings |
|---|---|---|---|---|
| HBM2 | 8 | 228 GB/s | 89% | DRAMsim3 stock |
| HBM3 | 16 | **736.6 GB/s** | 90% | JESD238 (Ramulator2) |
| HBM3e | 16 | **1072.1 GB/s** | 87% | HBM3 array + 9.6 Gbps PHY |

Scaling is **exactly linear** in channel count (verified at N = 1, 2, 4, 8, 16;
per-channel bandwidth constant to four significant figures).

> **These figures were corrected 2026-08-21 (was 707.8 / 1027.3).** gem5
> reports `bwRead::total` as bytes / TOTAL simulated time, and `smoke_hbm.py`
> appended a 1 us idle tail *inside* the measurement window, so every
> bandwidth number was diluted by 20/21 = **4.8%**. `--idle-ns` now defaults
> to 0; it is meaningful only for multi-burst duty-cycle experiments.
> Corroborated independently by the MoE replay cross-check, which implies
> 739.6 GB/s/stack (0.4% agreement). The check band was moved 40.0-51.2 ->
> 44.5-51.2 per channel: the old band **passed** the diluted 44.24 GB/s
> figure, because it had been set from that same measurement. A check
> calibrated from a flawed measurement cannot detect the flaw.

**736.6 GB/s/stack against the strongest external check available**: H100 is
specified at 3.35 TB/s across 5 HBM3 stacks = 670 GB/s/stack, at shipping
clocks below the top JEDEC bin. Nothing in the model was fitted to that
number.

Getting here required discarding an earlier, wrong result. Routing the same
traffic through a shared `NoncoherentXBar` produced apparent sub-linear scaling
that collapsed to 41% of peak at 16 channels (339 GB/s), and it was **not**
width or clock: 512 B @ 4 GHz and 2048 B @ 8 GHz both gave 330-339 GB/s. Nor
was it generator packet rate, nor row-buffer locality — gem5 strips interleave
bits before the memory sees the address, so each instance receives a compacted
contiguous stream whatever the interleave granularity, and varying that
granularity gave a non-monotonic 223-367 GB/s with no clean mechanism behind it.

Removing the crossbar resolves it completely, which localises the loss to gem5
XBar arbitration rather than to DRAMsim3 or the authored configs.

Two harness bugs were found on the way and are worth recording, because both
produced plausible-looking wrong answers rather than errors:

- Generators started at offsets aligned to the interleave stride, so they
  marched across instances in lockstep and hot-spotted one instance at a time
  (8.3 GB/s at 16 KB interleaving). Fixed by staggering each generator by one
  stride.
- In `--pairs` mode the generator span was sized from `fetch_bytes` while the
  memory range came from `mem_size`, so generators addressed outside their own
  memory. That one at least asserted rather than silently mis-measuring.

**Modelling consequence.** Model the HBM tier as **independent per-channel
controllers**, which is also what the hardware is — each HBM channel has its own
controller and datapath, and there is no monolithic crossbar between a GPU's L2
and its HBM channels. Do not put a single gem5 XBar in front of a whole stack;
it becomes the bottleneck and understates HBM by ~2x, in the direction that
flatters CXL.

**Sanity against shipping parts:** H100 (5 HBM3 stacks) is specified at
3.35 TB/s = 670 GB/s/stack, H200 (6 HBM3e) at 4.8 TB/s = 800 GB/s/stack. Both
sit *below* our full-rate figures, because shipping products clock HBM below
its maximum grade. When modelling a named GPU, set the data rate to that
product's effective rate rather than the JEDEC maximum, and say which was used.

## Verifying the authored configs

Bandwidth-vs-peak alone is a weak check: the "peak" is computed from the same
geometry that went into the file, so a wrong assumption moves the measurement
and the target together and still reads as ~90%. Worse, a saturating
**sequential** stream mostly exercises `tCCD` and the data bus — it leaves
`tRCD`, `tRP`, `tRAS`, `tFAW` (exactly the timings that were derived) almost
untested.

The sharp test is the **row-miss penalty**, measured with one outstanding
request so latency is observable, comparing random against sequential access:

| | row hit | row miss | penalty | `tRP + tRCD` |
|---|---|---|---|---|
| HBM3 | 18.0 ns | 51.2 ns | 33.2 ns | 35.6 ns |
| HBM3e | 17.8 ns | 52.0 ns | 34.2 ns | 35.6 ns |

The measured penalty tracks the specified `tRP + tRCD` to within ~7%; the
residual is random access occasionally landing on an already-open row across 32
banks, which dilutes the average downward. This is independent of the geometry arithmetic, so it
confirms DRAMsim3 parsed the derived cycle counts, that they convert back to
the intended nanoseconds, and that the derivation behaves as claimed.

Two corroborating observations:

- Row-hit latency falls 20.8 → 18.1 ns across generations, and the ~2.7 ns
  spread is exactly the burst shrinking (2 → 1.25 → 0.83 ns) while `CL` stays
  fixed at 14 ns.
- HBM3 and HBM3e deliver **identical** random-access bandwidth (15.2 GB/s),
  which is what "core timings held constant in ns" predicts — random access is
  core-bound, not interface-bound. Random drops to ~34% of sequential, the
  right magnitude for row-buffer thrashing.

**What is still unverified.** No comparison against real HBM3 silicon — the
absolute latencies (~45-49 ns loaded row-miss) are plausible for HBM3 but not
validated the way CXL-DMSim validated its CXL model against hardware. Refresh
behaviour, capacity edge cases and the power model are untested (the last is
irrelevant here). Most importantly, the per-channel-scaling assumption —
channels independent and evenly loaded under large sequential streams — is
argued, not measured, because the wrapper's request cap prevents testing the
full device. State it as an assumption in D1.

**Open item:** simulating a whole multi-stack GPU inside one gem5 memory system
would additionally need a crossbar carrying ~800 GB/s, which no plausible
`width x clock` delivers. Per-channel calibration plus analytic scaling avoids
this entirely; the alternative (many interleaved DRAMsim3 instances behind a
very wide bus) buys accuracy we do not need for bulk sequential streams.

## What was verified about the traffic generator

- `PyTrafficGen.start()` takes any iterable of **segments**, not packets.
  Exported: `createLinear / createRandom / createDram / createStrided /
  createTrace / createIdle / createExit`. The plan's "inject packets directly"
  is not available — but an expert fetch maps cleanly onto one `createLinear`
  burst and a compute window onto `createIdle`, so no protobuf is needed.
- **Segments are time-boxed, not work-boxed.** Once `data_limit` is reached,
  `linear_gen.cc` returns `MaxTick` and the segment idles out the remainder of
  its `duration`. If that idle exceeds `progress_check` (default 1 ms), gem5
  dies with "spent N ticks without making progress". Either size `duration` to
  the expected transfer time, or drop `data_limit` and let duration bound the
  burst.
- **`createIdle` does advance simulated time exactly** — verified: two 50 us
  windows plus two 5 us idles exited at precisely 110 us. This is what lets a
  trace express "layer n computes for X ns while layer n+1 prefetches".
- `min_period`/`max_period` are in ticks (1 tick = 1 ps). Sub-cycle values
  (e.g. 1) stall the generator; gem5's own example uses 500-1500.

## CXL Type-3 tier (`sim/configs/cxl_tier.py`)

    tgens -> aggbus -> link -> bridge x N -> DRAMsim3 x N
                       ^^^^    ^^^^^^^^^^
                    bandwidth  protocol latency + FIFOs

**Measured** (16 backend channels, so the link is the binding constraint):

| | nominal | achieved | % of link |
|---|---|---|---|
| local DDR x16 (baseline) | — | 99.1 GB/s | backend ceiling |
| CXL PCIe 4.0 x16 | 26 GB/s | 23.1 GB/s | **89%** |
| CXL PCIe 5.0 x16 (CXL 2.0) | 63 GB/s | **52.0 GB/s** | **83%** |
| CXL PCIe 6.0 x16 (CXL 3.0) | 121 GB/s | 64.5 GB/s | 53% (backend-bound) |

83% of link rate for CXL 2.0 sits on CXL-DMSim's silicon-measured 82-83% for
its ASIC part. The denominators differ (link rate vs local DDR bandwidth), so
treat the agreement as reassuring rather than as evidence.

Gen6 is backend-limited here, not link-limited: 121 GB/s exceeds what the
harness's DRAM tier sustains. Add channels before quoting a CXL 3.0 number.

**Latency**, single stream, one outstanding request:

| path | latency |
|---|---|
| local DDR (no protocol) | 22.2 ns |
| via CXL link + bridge | 186.9 ns |

The 164.7 ns delta is exactly `2 x 79 ns` — request and response each pay the
protocol cost — so the model reproduces its configured parameters. The absolute
figure is consistent with published CXL.mem load latency (250-400 ns) once the
CPU-side path both tiers share is added. The *ratio* (8.4x) is not comparable
to CXL-DMSim's 2.18x, which is measured against system-level DDR latency rather
than memory-port latency; adding ~60 ns of CPU-side path to both gives ~3.0x,
which matches the plan's table.

### Latency validation against CXL silicon

CXL-DMSim measured two real devices with LMbench (their Fig. 9, Table II) and
publishes the parameters it fitted to them (Table III). Their absolute numbers
are full load-to-use (including the LLC-miss path) while ours are memory-port,
so the comparable quantity is the **latency the CXL path adds**:

| | DDR-L | device | silicon adds |
|---|---|---|---|
| CXL-ASIC | 130 ns | 284 ns | **154 ns** |
| CXL-FPGA | 130 ns | 375 ns | **245 ns** |

Running both of their device configurations through our model — parameters
taken from Table III, not fitted:

| config | our delta | silicon | error |
|---|---|---|---|
| ASIC (`dev_proto_lat` 15 ns) | 161.2 ns | 154 ns | **+4.7%** |
| FPGA (`dev_proto_lat` 60 ns) | 255.5 ns | 245 ns | **+4.3%** |

Two devices differing by 4x in device-side protocol latency, both matched
within 5%, so the model tracks a *parameter change* correctly rather than
hitting one number by luck. The consistent +4.5% offset is the extra bus and
crossbar traversal in our topology, which is genuinely part of the path.

This also corrected `host_proto_lat` from 14 ns to **12 ns** (Table III) and
distinguished the host link FIFOs (128) from the device FIFOs (48); the device
is the binding constraint and is what the Bridge models.

### FloE validation (D1's external check)

FloE reports a Mixtral expert (>300 MB, FP16) taking **~15 ms over PCIe 4.0
x16**. At the measured 23.1 GB/s:

    352.3 MB / 23.1 GB/s = 15.2 ms      vs FloE ~15 ms   -> within 1.3%

Nothing was tuned to reach this.

### Three modelling traps found here

**gem5's XBar `width` is PER PORT, not shared.** Attaching N generators to a
"link" crossbar yields N x the intended bandwidth — it models a switch, not a
serial link. The first version returned an identical ~35.8 GB/s for 26, 63 and
121 GB/s links and had CXL *exceeding* local DDR. Fix: funnel every requestor
through a wide aggregation bus, then across a **single-port** stage whose one
layer serializes all traffic; only then is bandwidth `width x clock`.

**Backend channels must be interleaved.** With disjoint ranges a linear sweep
occupies one channel at a time, indistinguishable from a bandwidth ceiling.

**The baseline must share the CXL path's topology.** An earlier `--local-ddr`
bypassed the bridges entirely, so it had *less* outstanding capacity than the
CXL path (which gains 48 entries per bridge) and measured slower than CXL —
physically impossible. `--local-ddr` now builds the identical topology with the
link uncapped and protocol latency zeroed, so gem5's own arbitration and
outstanding-request limits affect both paths equally and cancel in the ratio.

## CXL model parameters (from SimCXL / CXL-DMSim)

SimCXL is a **fork of gem5 23.1**, not a plugin, so it cannot be combined with
vanilla 25.1. Its measured parameters are reusable regardless:

| component | parameter | value |
|---|---|---|
| host bridge | `bridge_lat` | 50 ns |
| host bridge | `proto_proc_lat` | 14 ns |
| device | `proto_proc_lat` | 15 ns |
| both | req/rsp FIFO depth | 48 |

≈79 ns of protocol overhead per direction on top of media latency, which is
what produces the paper's silicon-validated ~2.18x (ASIC) / ~2.88x (FPGA)
latency versus local DDR, and 82-83% of DDR bandwidth for the ASIC part.

## The link stage must be an EXACT-nominal ceiling

**A link stage of the wrong width measures the stage, not the link** — the
crossbar artifact from the HBM calibration, one level up. `cxl_tier.py` and
`moe_replay.py` build the link as a single-port XBar of
`width = round(link_gbps / bus_ghz)` bytes, and a 64 B packet occupies
`ceil(64 / width)` WHOLE cycles. So the achievable ceiling is
`64 / ceil(64/width) x bus_ghz`, which equals the nominal rate only when
**width divides 64 exactly** and `width x bus_ghz` equals the nominal:

| link | nominal | `--bus-ghz` | width | cyc/pkt | ceiling | |
|---|---|---|---|---|---|---|
| PCIe4 x16 | 26 | 4.0 | 6 | 11 | **23.3** | undershoots |
| PCIe4 x16 | 26 | **6.5** | 4 | 16 | **26.0** | exact |
| CXL2 x16 | 63 | 4.0 | 16 | 4 | 64.0 | +1.6%, non-binding |
| CXL3 x16 | 121 | 16.0 | 8 | 8 | **128.0** | OVERSHOOTS |
| CXL3 x16 | 121 | **15.125** | 8 | 8 | **121.0** | exact |

Both errors are silent and neither raises anything. The PCIe4 undershoot made
the "90% efficiency" figure a measurement of the crossbar; the CXL3 overshoot
let fetch traffic measure *faster than the link can physically run*. CXL2 was
never affected — width 16 divides 64 — which is why every CXL2 replay agreed
from the start. `check.py` now uses the exact clocks.

Re-measured at exact ceilings, steady-state effective bandwidth is
**PCIe4 25.33 GB/s (97.4%)** and **CXL3 106.6 GB/s (88%)** — the CXL3 figure
is essentially unchanged from the 105.6 measured at the 128 stage, i.e. it is
genuinely FIFO/protocol-bound rather than ceiling-bound.

## Device buffering is a modelled mechanism, not a caveat

Outstanding requests scale as bandwidth x round-trip, so a FIFO depth implies
a bandwidth cap by Little's law:

```
eff_link = min(fetch_rate, 8 bridges x entries x 64 B / RTT)
```

`analytical/trace_gen.py --fifo-entries` applies it, so the shallow-FIFO regime is
now *predicted from structure* instead of quoted as an assumption.

* **128 entries** — cap 234 GB/s at RTT 280 ns: inert above the 117 GB/s
  fetch rate, so deep-FIFO CXL3 is link-bound.
* **48 entries** (the silicon-validated CXL 2.0 ASIC depth) — cap
  **101.6 GB/s** at RTT **242 ns**, versus **101.4 GB/s** measured by
  bisection against gem5 replays with real 48-entry bridges. Replay ratios
  close from 0.865/0.884 to **0.990/1.013**.

The Little's-law RTT is *shallower* than the deep-FIFO loaded round trip
(242 vs 280 ns): a 48-entry window queues less, so it turns over faster.

**This supersedes the earlier "48 entries cap CXL3 at ~86 GB/s (71%)"
figure**, which was a SUSTAINED-traffic measurement. MoE expert fetch streams
reach ~101 GB/s (84%) on the same buffers, so shipping-depth CXL 3.0 silicon
is materially less punishing for this workload than we previously stated.

(One further harness note from the same work: the DDR5 ini needed the same
`trans_queue_size = 128` knee the HBM calibration found.)

## MoE decode replayed on gem5 (`sim/configs/moe_replay.py`)

The professor's framing: model rough GPU-type matmul on gem5 and use it as
the compute + memory requests from the host. For decode, a GEMV's memory
behaviour *is* the weight stream — every touched weight is read once per
step — so the host is generators that, per layer barrier, stream that
layer's HBM bytes (weights + KV) across the HBM channels and the missed
experts over the CXL path, with the next layer gated on both (plus an
analytic compute floor for compute-bound barriers). Both memory systems are
the calibrated ones, unchanged. The schedule comes from
`analytical/trace_gen.py --emit-schedule` (real routing logs, policy applied),
so each run doubles as a cross-check: gem5 decides all timing; the analytic
recurrence predicts it.

Co-simulation mechanics (each verified in isolation first):

* Python interleaves `m5.simulate()` with per-quantum generator restarts.
  A trace may only be re-started after its `createExit` fired — restarting
  mid-trace segfaults gem5 25.1 — and the exit must land strictly *before*
  the quantum boundary or it races the next `start()`.
* Progress is read live via `m5.stats.gem5stats.get_simstat` (no stats.txt
  churn); stream cursors advance by *measured* bytes, so backpressure is
  honoured exactly. `data_limit` caps each quantum so no stream overshoots
  its barrier target.
* Two gem5 patches were required.
  1. `src/cpu/testers/traffic_gen/base.cc`: `scheduleUpdate()` now uses
     `reschedule(..., true)` — a packet blocked on back-pressure at a restart
     boundary re-enters the scheduler after the new trace armed the update
     event, which asserts in vanilla. Behaviour-identical for stock
     single-start configurations (the assert would otherwise fire).
  2. `src/mem/dramsim3.cc`: `recvTimingReq()` now asks DRAMsim3 whether it
     will accept *this* transaction (`wrapper.canAccept(addr, isWrite)`) in
     addition to the aggregate `nbrOutstanding() < queueSize()` test.
     Vanilla flow control is a single counter against one queue capacity,
     but DRAMsim3 keeps a separate `read_queue_` and `write_buffer_` per
     channel, and gem5 drops a write from its counter as soon as it
     responds while DRAMsim3 holds it in `write_buffer_` until a watermark
     drain. gem5 then believes there is room, calls `enqueue()`, and
     DRAMsim3's own `assert(ok)` fires. Required by `--fill-writes` (below);
     provably a no-op for read-only traffic, where an outstanding read is
     held longer by gem5 than by DRAMsim3 and the aggregate test is already
     conservative — which is why ~100 read-only replay points never hit it.
* HBM channels are *sampled*: per-channel byte share is always that of the
  real 16-channel stack (exact under the measured linear channel scaling),
  default 4 instantiated → 4x fewer events.
* `--fill-writes` issues the cache-fill WRITE of every missed expert into
  HBM, mixed into the same generators as the reads (same channels, same
  rows) rather than aimed at a disjoint region. Off by default so the
  read-only timing campaign stays reproducible; on, the replay charges the
  same HBM traffic the analytic model does.
* Address cursors advance by **issued** bytes (generator `numPackets`), not
  completed bytes (controller counters). Advancing by completed bytes
  re-issues the in-flight window every quantum AND pins all fetch
  generators to the same span offset — under 256B interleave they then
  hammer one backend channel in lockstep, the same artifact the HBM
  calibration found, costing ~35% of link bandwidth (first CXL replay
  points ran at ratio ~1.5 because of exactly this; a synthetic 100 MB
  CXL-only barrier reproduces 50.7 GB/s within 2% after the fix).

Wall-clock limits what can be replayed: CXL traffic simulates at roughly
10 MB/s of modelled transfer, so validation points are truncated windows
(8–16 barriers) at operating points with bounded miss traffic, not full
decodes. Cross-validation results (replay vs analytic) live in docs/MODEL.md §5.

## Memory energy model

DRAMsim3 computes energy the standard way — IDD currents x VDD x time in each
state (the Micron power-calculation methodology, TN-40-07 lineage; Li et al.,
IEEE CAL 2020) — and writes per-channel totals to `dramsim3.json` when the
wrapper is destroyed. Two things had to be fixed before those numbers meant
anything:

**The unit trap.** DRAMsim3's energy stats are **V x mA x cycles, not pJ**.
Its energy increments carry timing in clock cycles and no tCK multiplication
happens anywhere in the energy path (`configuration.cc` / `simple_stats.cc`;
only `average_power` is unit-correct, because the cycles cancel there). True
pJ = reported x tCK(ns). At the stock DDR4 tCK = 0.63 ns the error is small
enough to miss; at our HBM3 tCK = 0.3125 ns it is 3.2x. The pre-fix "HBM3 =
3.885 pJ/bit vs published ~3.9" agreement was **this unit error landing on the
anchor by coincidence** — the true figure with the then-placeholder currents
was 1.21 pJ/bit. `measure_energy.py` and `check.py` now convert via each
config's own tCK.

**Calibrated-to-published HBM currents.** JESD238A fixes VDD (Table 70:
1.1 V) and defines the IDD states (§9) but leaves current *values* to vendor
datasheets — Table 90 is an empty template. So the HBM3/HBM3e `[power]`
sections keep the HBM2 template's state *ratios* and scale the whole current
family by one factor (HBM3 k = 3.270, HBM3e k = 4.648) such that the measured
sequential total equals the published HBM access energy, **3.97 pJ/bit**
(O'Connor et al., MICRO'17 — activation 1.21 + on-die movement 2.24 +
interposer I/O 0.3, +ECC). One target, one knob, stated as such. HBM3e is
held *equal* per bit — no vendor publishes HBM3E pJ/bit and Micron claims
better perf/W, so equal is conservative — and consequently **no
generation-relative HBM energy claims are made**. Stack-level sanity: 1.40
W/channel x 16 ch ≈ 22 W/stack at full bandwidth → ~112 W for an H100's five
stacks, consistent with 3.9 pJ/bit x 3.35 TB/s = 104 W.

**DDR5-6400 expander backend (authored, uncalibrated).**
`configs/dramsim3/DDR5_6400_4Gb_x8.ini` is one 32-bit DDR5-6400 subchannel authored
from gem5 25.1's `DDR5_6400_4x8` class, whose timings come from the Micron
DDR5 core datasheet and whose IDD currents come from the Micron 16Gb DDR5 die
rev A datasheet (DDR5-4800 grade, gem5's own convention). DRAMsim3 has no
DDR5 protocol enum, so it is expressed as `protocol = DDR4` (bank groups,
same refresh semantics) with DDR5 geometry, BL16, and DDR5 timings. Measured:
**11.85 pJ/bit** sequential — in the DDR/GDDR device-class range (O'Connor
Fig. 1a shows GDDR5 at 14) with *no* calibration, which is itself a check.
Known small understatements, documented: the DRAMPower-style activate term
goes slightly negative with these currents (IDD0 is measured cycling one bank
while IDD3N assumes all-bank active standby), and the gem5 class carries no
IPP/VPP current.

The expander backend is sized from production Type-3 briefs (papers/): 8
subchannels x 25.6 GB/s = 204.8 GB/s = Marvell Structera X 2504 ("4ch
DDR5-6400, 200 GB/s, <30 W typical"); Astera Leo runs 2ch DDR5-5600 per x16.
Both provision backend bandwidth above the x16 CXL link, so the **link
binds** — which the harness assumed and the briefs now justify. Energy runs
are single-instance (`--direct`) because multi-instance DRAMsim3 runs collide
on the single `dramsim3.json`; per-channel energy is identical under
symmetric interleave, so totals scale by channel count.

E_cxl composition: DRAMsim3 measures only E_dram. The SerDes + controller
terms exist in **no** DRAM simulator and are analytic, bracketed by
`system_params.py` ENERGY entries (published PCIe5 PHY 11.4 pJ/bit; whole
expander device ceiling 30 W / 200 GB/s = 18.75 pJ/bit) and swept 2–19
pJ/bit.

## Files

| path | role |
|---|---|
| `gem5/` | vanilla gem5 25.1.0.1 + the `cstdint` fix |
| `gem5/ext/dramsim3/DRAMsim3/` | DRAMsim3, built |
| `gem5/ext/dramsim3/DRAMsim3/configs/DDR5_6400_4Gb_x8.ini` | authored DDR5-6400 subchannel (timing+IDD from Micron via gem5) |
| `SimCXL/` | CXL-DMSim, gem5 23.1 fork — reference for the CXL model |
| `sim/configs/smoke_hbm.py` | calibration harness (`--direct`, `--xbar-width`, `--replicate-example`) |
| `sim/configs/cxl_tier.py` | CXL tier; DDR5-6400 backend, Structera-consistent |
| `measure_energy.py` | pJ/bit measurement (tCK unit conversion lives here) |
| `check.py` | the validation suite — `make check` |
| `build-env.sh` | conda paths for zlib/protobuf/m4 |
