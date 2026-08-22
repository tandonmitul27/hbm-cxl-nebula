# MODEL.md — HBM+CXL memory model for MoE inference

The model card: what is modelled, where every number comes from, how it is
verified, and what it cannot claim. Someone with this file, `sim/README.md`
and `sim/system_params.py` should be able to rebuild the model and reproduce
every validation with the commands given.

## 1. System modelled

Per the project clarification: the **GPU is the CXL host**. One hop.

```
            ┌───────────────┐
            │      GPU      │  compute: roofline (analytical/gpu_model.py)
            │  (H100 anchor)│  T_layer = max(bytes/BW_hbm, FLOPs/FLOPS_peak)
            └───┬───────┬───┘
     on-package │       │ CXL 2.0/3.0 over PCIe5/6 electricals
        ┌───────┴──┐   ┌┴──────────────────────────────┐
        │   HBM3/3e │   │ CXL Type-3 memory expander    │
        │ 5x stacks │   │  bridge 50ns + host proto 12ns │
        │ 16ch x64b │   │  + dev proto 15ns, dev FIFO 48 │
        │ (JESD238A)│   │  backend: 8x DDR5-6400 subch   │
        └───────────┘   │  (= Marvell Structera X 2504)  │
                        └────────────────────────────────┘
```

HBM-only and HBM+CXL are **separate models run individually** (deliverable
framing), sharing the same calibrated components. Data flow for tiering: CXL
holds the canonical home of every routed expert; HBM holds pinned weights
(attention/shared/embeddings) plus a policy-managed expert cache; a fill
crosses the link once and is written into HBM (both costs accounted). No
pooling, no multi-host, no CXL switch (its +100 ns is parameterised but
unused).

This is the forward-looking single-hop topology; shipping hardware attaches
CXL to a CPU (a second PCIe traversal away from the GPU). Stated wherever
results are quoted.

## 2. Components and provenance

Full table with values: `python sim/system_params.py`. Summary:

| component | source | confidence |
|---|---|---|
| Link peak rates (26/63/121 GB/s) | PCIe 5.0/6.0 spec arithmetic | spec |
| Effective link bandwidth, MoE fetch (24.9/50.7/114.5) | measured at exact-nominal link stages, fitted by bisection vs the gem5 replay campaign (§5b) | measured |
| HBM geometry, voltages, PC-scoped timings | JESD238A (we hold the PDF) | spec |
| HBM array timing values | Ramulator2 JESD238-cited presets (vendor-representative; JEDEC leaves values to vendors by design) | datasheet-class |
| HBM3e array timings | held from HBM3, PHY scaled | ESTIMATE (declared) |
| CXL protocol path (50+12+15 ns, FIFO 48) | CXL-DMSim Table III, silicon-validated | datasheet |
| Expander backend (8x DDR5-6400 subch) | Marvell Structera X 2504 / Astera Leo briefs (papers/) | datasheet |
| DDR5 timings + IDD | Micron datasheets via gem5 25.1 `DDR5_6400_4x8` | datasheet |
| HBM energy | calibrated to O'Connor MICRO'17 3.97 pJ/bit (one scale factor; method stated) | calibrated |
| CXL link+ctrl energy | bracketed [2, 19] pJ/bit: PCIe5 PHY 11.4 (CICC'20), device ceiling 18.75 (Structera <30 W / 200 GB/s); **swept, never fixed** | bracketed |
| GPU FLOPS / stacks / aggregate BW | NVIDIA H100/H200/B200 datasheets | datasheet |
| GPU-side CXL lane count (x16) | no shipping GPU exposes CXL host ports | ESTIMATE (swept) |

## 3. Timing model (analytical/trace_gen.py)

Exact barrier recurrence over the measured routing logs (17.1M routing
records, 5 models — `data/README.md`):

```
for each (step, layer) barrier:
    misses  = batch-union experts not resident under the policy
    issue   = layer_begin[idx - depth]        # prefetch lookahead
    arrive  = max(issue, link_free) + RTT + miss_bytes/BW_link
                                    + miss_bytes/BW_hbm
    # per-expert arrival gating: RESIDENT experts stream and compute while
    # the missed one is still in flight; only the missed expert's own HBM
    # read waits for it to land.
    now     = max( max(now + t_resident, arrive) + t_missed,
                   now + compute_floor )
```

* `RTT` (~280 ns loaded, per barrier fetch: measured CXL added latency +
  expander first access) is carried so the model demonstrates
  latency-insensitivity at 12-336 MiB transfers (<0.01%) rather than
  assuming it.
* `T_layer = t0 + max(bytes/BW, FLOPs/FLOPS_peak)`: `t0` is per-barrier
  launch/dispatch/sync overhead, bracketed [5, 40] us (default 20; CUDA
  launch x kernels/layer, CUDA-graph amortisation -- `system_params.py
  T0_LAYER`), never fitted: the Phase-1 RTX 4070 timings are inflated ~5x
  by transformers' Python expert loop, so they serve as an upper-bound
  check only.  Prefill FLOPs include the causal attention-score quadratic
  (`gpu_model.attn_quad_flops`); decode's quadratic is charged as KV
  memory traffic instead, which is where its cost actually is.

* `link_free` serialises transfers: prefetching cannot conjure bandwidth.
* Prefetch depth costs capacity: `slots -= depth x top_k` (staged experts
  occupy HBM while in flight).
* Fills pay HBM write bandwidth (certified: 90/10 mix keeps 95% of read-only
  throughput).
* Decode: `T_layer = bytes/BW` (bandwidth-bound; intensity ~B « ridge 295).
  Prefill: one barrier per layer over the whole prompt, compute-bound,
  `T_layer = 2 x params x B x P / FLOPS_dense`. Both phases run from real
  routing logs; fp16 and fp8 via `--dtype`.
* Policy ladder: none / static-popular / LRU (deliberately pathological
  under cyclic layer access) / LRU-per-layer / Belady oracle — floor and
  ceiling bound every result.

The recurrence is exact arithmetic; its **inputs** are certified by the
check suite (§5) and the recurrence itself is exercised end-to-end by the
gem5 replay (§5b) at 103 operating points, 98 within ±3%.

## 4. Energy model (analytical/energy_model.py)

```
E = bytes_hbm_read x e_hbm_rd + bytes_hbm_fill x e_hbm_wr
  + bytes_cxl_read x (e_ddr5_rd + e_link_ctrl)
  + P_bg(u_hbm, u_ddr5) x T          ;  EDP = E x T

P_bg(u) = sum over devices of  P_ref + u x P_act_stb + (1 - u) x P_pre_stb
```

Dynamic pJ/bit and background watts are the component split of the
calibrated DRAMsim3 runs (activate+column vs refresh+standby), so idle time
is charged as power x time and EDP penalises slow configurations twice — by
construction.

**Background power is state-resolved.** A DRAM device does not draw one
number: it sits in *active* standby while a bank is open and *precharge*
standby when none is, and the two differ by ~35% on HBM3 and ~55% on DDR5.
The recurrence already tracks how long each device streams, so it reports
duty cycles `u_hbm`, `u_ddr5` and the energy model interpolates between the
two measured endpoints rather than billing the saturated one for the whole
window. This matters precisely in the regime the tiering results occupy: a
stall-dominated run leaves HBM idle most of the window, and charging it
active standby throughout over-billed background by ~13%. The per-state
coefficients (`p_act_stb`, `p_pre_stb`, `p_ref` in `energy_model.py`) are
measured from each config's own per-state cycle counters — HBM3, HBM3e and
DDR5 each independently, never scaled from one another. `e_link_ctrl` is swept 2–19 pJ/bit (cited bracket); every
energy conclusion is reported across the sweep. **Unit trap fixed on the
way**: DRAMsim3 emits V·mA·cycles, not pJ — multiply by tCK; the pre-fix
"3.885 pJ/bit ≈ 3.9 published" agreement was that unit error landing on the
anchor by coincidence (sim/README.md, "Memory energy model").

Compute-side energy is out of scope (professor: memory side only).

## 5. Verification — `make check` (22 checks)

Every number the model's credibility rests on, one command, tolerances
stated, exit nonzero on any failure. `make check-fast` skips the two
full-stack runs (20/22 in ~2 s).

| group | checks | anchor |
|---|---|---|
| address maps | 4 models vs downloaded checkpoint sizes, ±1% | checkpoints on disk |
| routing logs | structural integrity, 20 files, 17.1M records | phase1/check_integrity.py |
| row-miss penalty | HBM3, HBM3e ≈ configured tRP+tRCD | config self-consistency |
| bandwidth | single channel 46.04; HBM3 stack 736.6 vs H100 670; HBM3e 1072.1 | NVIDIA datasheet |
| CXL latency | ASIC +159.7 vs silicon 154; FPGA +248.5 vs 245 | CXL-DMSim Table II |
| CXL bandwidth | FloE Mixtral fetch ~13.9 vs ~15 ms; CXL3 x16 106.6 GB/s | FloE paper; spec |
| fill mix | 90/10 read/write keeps 95% of read-only BW | fill accounting |
| energy | HBM3 3.97, HBM3e 3.97 (calibrated), DDR5 11.8 (uncalibrated, in class range) | O'Connor MICRO'17 |
| recurrence invariants | `total_s >= compute_s` at every operating point; `t0` adds exactly 20 us per barrier | pure arithmetic, no gem5 (~2 s) |

## 5b. Decode replayed ON gem5 (`sim/configs/moe_replay.py`)

Per the project requirement, GPU-type matmul is modelled on gem5 as the
compute + memory requests from the host: for decode, a GEMV's memory
behaviour *is* its weight stream, so per layer barrier the host streams the
layer's HBM bytes (weights + KV) across the HBM channels and the missed
experts over the CXL path; the next layer gates on both plus an analytic
compute floor. Schedules come from the real routing logs with the policy
applied (`trace_gen.py --emit-schedule`), so every replay doubles as a
cross-check of the analytic recurrence — gem5 decides all timing, the
recurrence predicts it. Mechanics, and the two local gem5 patches the
co-simulation needed, are documented in sim/README.md ("MoE decode
replayed on gem5").

### Validation campaign: 103 gem5 replay points

The recurrence is validated against gem5 at **103 operating points**: mean
ratio (replay/analytic) **0.998**, stdev 0.011, **97 of 103 within +/-3%**.
Per-point data is frozen in `docs/replay_points.csv`; `docs/confidence_analysis.py`
regenerates every figure below and in docs/VALIDATION_CAMPAIGN.md.

| axis | n | ratio range | mean |
|---|---|---|---|
| prefill | 23 | 0.973 – 1.005 | 0.993 |
| static, CXL2 (models x capacity x batch) | 15 | 0.992 – 1.005 | 0.998 |
| B200 / 8 stacks of HBM3e | 12 | 1.008 – 1.039 | 1.016 |
| dynamic policies | 11 | 0.992 – 1.001 | 0.993 |
| link + FIFO refits | 10 | 0.990 – 1.016 | 1.003 |
| capacity ladder | 9 | 0.991 – 0.994 | 0.992 |
| high residency / all-hit | 6 | 0.968 – 1.023 | 0.999 |
| batch union | 5 | 0.992 – 0.992 | 0.992 |
| expert granularity | 4 | 0.992 – 0.994 | 0.992 |
| dtype fp8 | 3 | 0.991 – 0.993 | 0.992 |
| long horizon (96–1024 barriers) | 3 | 0.992 – 0.993 | 0.993 |
| H200 / 6 stacks of HBM3e | 2 | 1.004 – 1.004 | 1.004 |
| **all** | **103** | **0.968 – 1.039** | **0.998** |

These figures are **uncorrected**. The replay's quantum sampling is a real
measured artifact, but the corpus-wide fit for it no longer reproduces its own
controlled probe at 103 points (0.256 against 0.666) and is therefore not
subtracted -- see docs/VALIDATION_CAMPAIGN.md. It changes little: correcting
moves the mean 0.9979 -> 0.9963 and stdev 0.0113 -> 0.0103. The 95%
prediction interval is **+/-2.4%**: decode n=80 at 0.9992, prefill n=23 at
0.9934 (23/23 in band), dynamic n=11 at 0.9930.

**Convention.** Ratios are t0-free. `t0` is exact serial arithmetic on both
sides, but it is serial in the analytic model while the replay's gate is
`max(stream, floor)`, which hides it whenever streaming dominates -- leaving
it in measures bookkeeping, not the memory recurrence. Five points were
re-run on gem5 with `--t0-us 0` to audit the conversion; they reproduce the
offline recomputation to within **0.002** (5/5).

#### Known residual: a +1.5% offset on HBM3e systems

The one structure left in the residual tracks **stack count**, and nothing
else:

| memory system | n | mean | sd |
|---|---|---|---|
| HBM3 x5 (3683 GB/s) | 88 | 0.9952 | 0.0090 |
| HBM3e x6 (6433 GB/s) | 3 | 1.0041 | 0.0002 |
| HBM3e x8 (8577 GB/s) | 12 | 1.0157 | 0.0118 |

It is not the quantum artifact in disguise: the correlation between the HBM3e
indicator and `quantum/t_barrier` across the corpus is **+0.023**.

H200 and B200 share a per-stack constant and a DRAMsim3 config, so a wrong
device constant cannot open a gap between them -- only something scaling with
stack count can. Three candidate explanations were tested and **none holds**:

1. *Serial placement of the cache-fill write.* The recurrence charges
   `nb/hbm` on the arrival path; a paired replay with `--fill-writes` on and
   off shows the fill costs **zero** time (identical `simSeconds`, with
   gem5's counters confirming the writes occurred). But removing the term
   raises the overall mean to 1.0123 and *widens* stdev to 0.0143, and it
   requires link constants 0.8–3% below their independently measured values
   (CXL3 would need 111.1 against a measured 114.5). The term is empirically
   load-bearing; it is kept, and it is not literally a serial fill.
2. *A per-fetch overhead.* Ruled out: `xfer_lat` at 3 us, ten times the
   measured value, moves the residual by 0.001.
3. *Channel-sampling burst length.* Not tested -- a scan over
   `--hbm-channels` returns bit-identical ratios **by construction**, since
   the per-channel share is `hbm_bytes / STACKS / 16` and does not depend on
   how many channels are instantiated.

So it is reported, not explained: **quote HBM3e-system results with a
+1.5% bias**. It sits inside the +/-2.4% prediction interval, it does not
reorder anything, and every headline result is a capacity cliff of ~11x
rather than a percent-scale timing difference.

**Convention.** Ratios are t0-free. `t0` is exact serial arithmetic on both
sides, but it is serial in the analytic model while the replay's gate is
`max(stream, floor)`, which hides it whenever streaming dominates -- leaving
it in measures bookkeeping, not the memory recurrence. Five points were
re-run on gem5 with `--t0-us 0` to audit the conversion; they reproduce the
offline recomputation to within **0.002** (5/5).

#### Structural change the campaign forced: per-expert arrival gating

A layer's **resident** experts are already in HBM, so their weight reads and
GEMVs proceed *while* a missed expert is still in flight; only the missed
expert's own read waits for it to land:

```
end = max( max(begin + t_resident, arrive) + t_missed,  begin + floor )
```

The earlier recurrence serialised the whole layer behind the fetch, which
over-predicts by up to one layer-time per missing barrier. That is ~1% in
decode (compute is tens of us against ms-scale fetches) but **up to 10% in
prefill**, where a barrier's compute (~0.8 ms) is comparable to a fetch.

It was found because prefill had **no** replay point until this campaign; the
first one came in at 0.912. The mechanism predicts the error should scale
with residency, and it does, monotonically over 10 points:

| prefill hit rate | 99.2% | 98.4% | 98.4% | 81.5% | 49.3% | 37.5% | 19.4% |
|---|---|---|---|---|---|---|---|
| serial-model ratio | 0.898 | 0.912 | 0.925 | 0.972 | 0.973 | 0.992 | 0.977 |

Gating improves **every** regime at once, with no fitted parameter:

| | serial | gated |
|---|---|---|
| decode, static (48) | 0.9924 | **1.0000** |
| decode, dynamic (7) | 0.9864 | **0.9963** |
| prefill (10) | 0.9597, 7/10 in band | **0.9905, 10/10** |

A blanket "everything overlaps" variant was tested and **rejected**: it fixes
prefill but breaks decode (42/42 -> 31/42 in band). Only the per-expert form
works in both.

#### Harness and constant corrections

1. **The link stage must be an exact-nominal ceiling.** A single-port XBar of
   `width = round(rate/clock)` carries a 64 B packet in `ceil(64/width)`
   WHOLE cycles, so unless `width` divides 64 the harness measures itself.
   PCIe4 at the old clock capped at 23.3 GB/s (undershoot); CXL3 at 128 GB/s
   (**overshoot** -- above the FLIT-corrected nominal, letting fetch traffic
   appear faster than the link can physically run). CXL2 was never affected
   (width 16 divides 64), which is why its points agreed from the first run.
   `sim/check.py` now pins exact clocks; the rule is tabulated in
   sim/README.md.
2. **Link constants re-measured** at exact ceilings: PCIe4 **23.5 -> 24.9**,
   CXL3 **105.6 -> 114.5** for fetch traffic (105.6 retained and relabelled
   as the steady-state figure). The CXL3 value was fitted *twice*: 117.0
   under the serial recurrence (4 points, spread 115.2-121.0), then 114.5
   under gating (15 points, spread **113.8-114.9**). The better model
   tightened its own calibration 5x -- independent support for gating.
3. **Device buffering is a modelled mechanism**, not an assumption:
   `eff_link = min(fetch_rate, 8 x entries x 64 B / RTT)`. At 48 entries it
   predicts 101.6 GB/s against 101.4 measured. This **supersedes the earlier
   "48 entries cap CXL3 at ~86 GB/s (71%)"** claim, which was a
   sustained-traffic measurement -- fetch streams reach ~101 GB/s (84%) on
   the same buffers.

#### Resolved: the all-hit residual was two bugs, not a residual

An earlier draft of this section reported a "known residual" -- all-hit
barrier ratios correlating with per-barrier stream time at r = -0.991, from
1.023 at 39 us/barrier to 0.968 at 223 us -- and argued it should be
documented rather than modelled, on the grounds that fitting two parameters
to a <=3.2% effect would trade a structural model for a curve fit.

That was wrong. Probing it (vary `--quantum-ns` and `--hbm-channels` on
fixed schedules) decomposed it into two separate causes, both fixable:

**(a) Replay quantum sampling -- a harness artifact, correctly NOT modelled.**
`moe_replay.py` only tests barrier completion at quantum boundaries, so each
barrier's end rounds up. Across 6 probe points spanning two models and three
quanta:

```
ratio = 0.9556 + 0.666 x (quantum / t_barrier)      R^2 = 0.99997
```

Changing only `--quantum-ns` (4000 -> 16000) moved a ratio from 1.023 to
**1.228**. This term accounts for ~92% of the model-to-model spread, and it
is a property of the measurement harness, not of the memory system: fitting
the analytic model to it would have made the model WORSE at predicting real
hardware while making the validation numbers look better. Validation points
should keep `quantum / t_barrier` small; the campaign's stall-dominated
points (ms-scale barriers) are unaffected at <0.3%.

**(b) A diluted bandwidth calibration -- a real bug, now fixed.**
With (a) removed, a uniform 4.3% offset remained, identical across models
and invariant to `--hbm-channels` 4/8/16 (so not a sampling artifact).
Cause: gem5 reports `bwRead::total` as bytes / TOTAL simulated time, and
`smoke_hbm.py` appended a 1 us idle tail inside the measurement window, so
every HBM bandwidth figure was low by 20/21 = **4.8%**.

| | was | is |
|---|---|---|
| HBM3 stack | 707.8 | **736.6** GB/s |
| HBM3E stack | 1027.3 | **1072.1** GB/s |

Confirmed two ways: re-measuring with `--idle-ns 0` gives 736.6, and the
replay independently implies 739.6 (0.4% agreement). `--idle-ns` now
defaults to 0.

**The check suite could not have caught this**: its acceptance band
(40.0-51.2 GB/s) was set from the same diluted measurement, so the diluted
44.24 GB/s passed. A check calibrated from a flawed measurement confirms the
flaw. The band is now 44.5-51.2, and `check_recurrence_invariants()` adds
arithmetic invariants that hold regardless of any measured constant.

**Effect on agreement.** With the corrected constant the campaign is
unbiased: mean **0.9979** over 103 points, stdev 0.0113, 98/103 within +/-3%.

**On subtracting the quantum term.** The replay's quantum sampling is a
measured harness artifact -- a controlled probe varying only `--quantum-ns`
gives `ratio = 0.9556 + 0.666 x (quantum/t_barrier)`, R^2 = 0.99997. At 68
points a least-squares fit over the whole corpus returned 0.699, agreeing
with the probe, and that agreement is what licensed subtracting it. At 103
points the corpus fit is **0.256** and no longer reproduces its own
controlled measurement, so it is no longer subtracted from any published
figure. The artifact is still real and still must not be modelled; it simply
cannot be removed by one corpus-wide slope. The cost of not removing it is
small: mean 0.9979 vs 0.9963, stdev 0.0113 vs 0.0103.

#### Coverage

All 4 MoE models; expert granularity 12-336 MiB; capacity 2.5-100% plus each
GPU's native HBM anchor; batch 1/4/16/64; fp16 and fp8; PCIe4, CXL2 and CXL3;
48- and 128-entry FIFOs; **all three GPUs in the sweep** (H100/HBM3 x5,
H200/HBM3e x6, B200/HBM3e x8 -- no row of `sweep.parquet` is extrapolation
along the GPU axis); decode and prefill; static and 4 dynamic policies;
windows of 8-1024 barriers. The headline operating point
(Mixtral b16 on 80 GiB H100, 91% resident) replays at **0.990** over 128
barriers. **No drift over a 64x horizon range**: OLMoE static reads 0.985 at
16 barriers, 0.983 at 256 and 0.984 at 1024 -- consistent with the recurrence
being a sum of independently-correct terms rather than an iterated map that
accumulates error.

#### Composed energy, validated end to end

Energy was previously validated **per device** (HBM pJ/bit vs O'Connor, DDR5
vs the device class) but never as a composition. `moe_replay.py --energy-dir`
gives each DRAMsim3 instance its own output directory -- they previously
overwrote one shared `dramsim3.json` -- so a full replay reports HBM +
expander energy together. 20 runs across both HBM generations, both dtypes,
duty cycles from 0.02 to 1.00, with cache-fill writes on and off:

| | n | dynamic | background | composite |
|---|---|---|---|---|
| read-only | 10 | 0.976 – 1.000 | 0.982 – 1.006 | 0.992 – 1.001 |
| with `--fill-writes` | 10 | 0.989 – 1.000 | 0.982 – 1.008 | **0.996 – 1.005** |

Buckets follow the *model's* split, not DRAMsim3's: refresh is billed as
background here because that is where the model puts it. Grouping it with the
dynamic terms (DRAMsim3's own grouping) leaves the total right and makes
every per-component ratio meaningless.

The sharpest result is in the background term, split by duty cycle:

| regime | n | gem5 / model |
|---|---|---|
| stall-dominated (`u_hbm` < 0.5) | 14 | **1.002** |
| saturated (`u_hbm` >= 0.5) | 6 | 0.983 |

State-resolved background is *exact* where the tiering results live, and 1.7%
conservative at the saturated end. The superseded saturated-endpoint model
billed `0.252 W/channel x wall-time` regardless of activity and over-charged
by ~13% in exactly the stall-dominated regime -- biasing absolute Joules and
EDP upward where it mattered most.

Turning on fill-writes also tightens the dynamic term (mean 0.987 -> 0.994,
range 0.976–1.000 -> 0.989–1.000), the expected direction if the model was
right and the read-only replay was the incomplete one.

#### Remaining gaps

The CXL link/controller energy term (2–19 pJ/bit) has no gem5 counterpart and
is excluded from the comparison above -- only real silicon can settle it. It
is the dominant uncertainty in every energy conclusion (x1.40 end to end).

## 6. Known limitations (state these with results)

1. **Single-hop GPU-host topology** is forward-looking; no shipping GPU
   exposes CXL host ports. Lane count x16 is assumed and swept.
2. **HBM IDD currents are calibrated, not vendor data** (one scale factor to
   one published target; JEDEC leaves the values to vendor NDAs). HBM3e is
   held equal per bit; no generation-relative HBM energy claims. The
   *per-state standby* coefficients are separate: each generation's are
   measured from its own DRAMsim3 cycle counters, never scaled from another
   generation's.
3. **CXL 3.0 buffering is modelled, but no CXL 3.0 silicon exists to
   validate the device against.** The Little's-law cap (§5b) is validated
   against gem5 at both depths, and at 48 entries (shipping CXL 2.0 ASIC
   depth) fetch streams reach ~101 GB/s (84%) rather than the ~86 GB/s (71%)
   previously quoted from a sustained-traffic measurement. What remains
   unvalidated is the *device model itself* at 3.0 rates: the protocol
   latencies are CXL 2.0 silicon figures assumed to carry over.
4. **e_link+e_ctrl split is not public** — bracketed and swept, never fixed.
5. **GPU compute is a roofline plus a bracketed overhead**: dense datasheet
   FLOPS; per-barrier launch overhead t0 is an ESTIMATE swept [5, 40] us,
   not a measurement. That bracket moves an all-resident result by 1.16x and
   a stall-dominated one by 1.00x — it is the dominant uncertainty exactly
   where nothing misses, and irrelevant where the tiering conclusions live. Prefill carries the attention-score quadratic;
   decode's quadratic is represented as KV traffic, not FLOPs.
6. **KV-cache traffic is modelled to first order** (per-barrier reads grow
   with sequence length; the fully-grown pool is reserved out of HBM
   capacity; MLA modelled compressed, as the architecture intends).
   Prefill's quadratic attention FLOPs are now modelled; its quadratic
   score/probability *reads* remain out, and KV layout is not part of the
   address map.
7. Routing logs come from batch ≤ 64 on the profiling GPU; larger batches
   would union more experts per barrier (direction: worse for tiering).
8. DDR5 activate-energy term clamps a small negative artifact of the
   DRAMPower-style formula; IPP/VPP current not carried by the gem5 class
   (both understate expander energy slightly).

## 7. Reproduce

```
make check                     # all 22 validations, ~6 min
python sim/system_params.py    # every parameter + provenance
python sim/measure_energy.py --config HBM3_16Gb_x64_1ch   # pJ/bit breakdown
python analytical/trace_gen.py --tag Mixtral-8x7B --batch 16 \
    --hbm-gib 24 --policy lru_layer --phase decode --dtype fp16
```
