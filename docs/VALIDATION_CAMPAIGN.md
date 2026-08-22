# Validation campaign — durable record

103 gem5 replay points validating `analytical/trace_gen.py` against
`sim/configs/moe_replay.py`. Per-point data is frozen in
`docs/replay_points.csv`; `docs/confidence_analysis.py` regenerates every
number below from that file plus `results/sweep.parquet`, with no dependency
on the campaign working directory.

## Confidence, split three ways

Conflating these overstates the weakest, so they are reported separately.

### A. Accuracy where measured (replay / analytic, **uncorrected**)

| regime | n | mean | 95% CI on mean | within ±3% | 95% prediction interval |
|---|---|---|---|---|---|
| all | 103 | 0.9979 | [0.9958, 1.0001] | 98/103 | ±2.4% |
| decode | 80 | 0.9992 | [0.9967, 1.0018] | 75/80 | ±2.4% |
| prefill | 23 | 0.9934 | [0.9899, 0.9966] | 23/23 | ±2.3% |
| static | 92 | 0.9985 | [0.9961, 1.0009] | 87/92 | ±2.5% |
| dynamic | 11 | 0.9930 | [0.9921, 0.9946] | 11/11 | ±1.2% |
| H100 / HBM3 ×5 | 88 | 0.9952 | [0.9934, 0.9971] | 86/88 | ±2.2% |
| H200 / HBM3e ×6 | 3 | 1.0041 | [1.0039, 1.0043] | 3/3 | ±0.5% |
| B200 / HBM3e ×8 | 12 | 1.0157 | [1.0097, 1.0227] | 9/12 | ±3.9% |

**These are uncorrected on purpose.** The replay's quantum sampling is a real,
measured harness artifact — a controlled probe varying only `--quantum-ns`
gives `ratio = 0.9556 + 0.666 × (quantum/t_barrier)`, R² = 0.99997. At 68
points a least-squares fit over the whole corpus returned 0.699 and
cross-validated against that probe, which is what licensed applying it. At 103
points the corpus fit is **0.256** and no longer reproduces its own controlled
measurement, so it is reported for comparison and not used as the claim. It
matters little either way: correcting moves the mean 0.9979 → 0.9963 and sd
0.0113 → 0.0103.

**Claim that is defensible:** a single latency prediction lands within ~2.4%
of gem5, and the model is unbiased to within 0.3%. HBM3e systems carry a
documented +1.5% offset (below).

### B. Sensitivity to values that are bracketed, not measured

These bound what calibration *cannot* pin down, and for energy they dominate
the accuracy figure above.

| parameter | range | effect |
|---|---|---|
| `e_link` | 2–19 pJ/bit | energy ×1.13 median, **×1.40 max** |
| `t0` | 5–40 µs | **×1.16 all-resident**, ×1.00 stall-dominated |

`t0` uncertainty matters *only* where nothing misses — exactly where it
becomes the dominant term. Every energy conclusion must be quoted across the
`e_link` bracket.

### C. Coverage — where the accuracy claim transfers

| axis | validated |
|---|---|
| models | 4/4 |
| links | 3/3 (24.9, 50.7, 114.5 GB/s) |
| batches | 4/4 (1, 4, 16, 64) |
| dtypes | 2/2 |
| phases | 2/2 (decode n=80, prefill n=23) |
| policies | 5/5 (static + 4 dynamic) |
| horizon | 8–1024 barriers, no drift |
| GPUs | 3/3 — H100, H200, B200 all replayed |
| stack counts | 5, 6, 8 |

**No row of `sweep.parquet` is extrapolation along any swept axis.**

## Known residual: a +1.5% offset on HBM3e systems

The one structure left in the residual tracks stack count and nothing else:
HBM3 ×5 at 0.9952, HBM3e ×6 at 1.0041, HBM3e ×8 at 1.0157. It is not the
quantum artifact in disguise — the correlation between the HBM3e indicator
and `quantum/t_barrier` is **+0.023**. H200 and B200
share a per-stack constant and a DRAMsim3 config, so a wrong device constant
cannot open a gap between them.

Three explanations were tested; none holds:

1. **Serial placement of the cache-fill write.** A paired replay with
   `--fill-writes` on and off shows the fill costs *zero* time — identical
   `simSeconds`, with gem5's counters confirming 4.57 MB actually written per
   sampled channel. But removing the analytic's `nb/hbm` arrival term raises
   the overall mean to 1.0123, widens stdev to 0.0143, and demands link
   constants 0.8–3% below their independently measured values (CXL3 would
   need 111.1 against a measured 114.5). Kept: empirically load-bearing, and
   not literally a serial fill.
2. **A per-fetch overhead.** Ruled out — `xfer_lat` at 3 µs, ten times the
   measured value, moves the residual by 0.001.
3. **Channel-sampling burst length.** *Not* tested. A scan over
   `--hbm-channels` returns bit-identical ratios by construction: the
   per-channel share is `hbm_bytes / STACKS / 16` and does not depend on how
   many channels are instantiated. The scan proves nothing and should not be
   cited as a negative result.

Reported, not explained. It sits inside the ±2.4% prediction interval,
reorders nothing, and every headline result is a capacity cliff of ~11×
rather than a percent-scale timing difference.

## Composed energy

20 runs, both HBM generations, both dtypes, duty cycles 0.02–1.00:

| | n | dynamic | background | composite |
|---|---|---|---|---|
| read-only | 10 | 0.976 – 1.000 | 0.982 – 1.006 | 0.992 – 1.001 |
| with `--fill-writes` | 10 | 0.989 – 1.000 | 0.982 – 1.008 | 0.996 – 1.005 |

Background term by duty cycle: **1.002** stall-dominated (n=14), 0.983
saturated (n=6). State-resolved background is exact where the tiering results
live; the superseded saturated-endpoint model over-charged by ~13% there.

Buckets follow the *model's* split, not DRAMsim3's — refresh is background.
Grouping it with the dynamic terms leaves the total right and makes every
per-component ratio meaningless.

## Known limits that are NOT holes to fill

* **`e_link` bracket** — no gem5 counterpart, no vendor PHY/controller split
  published. Only real silicon settles it.
* **Single-hop GPU-host topology** — no shipping GPU exposes CXL host ports.
  Forward-looking by construction; the optimistic case for CXL.
* **CXL 3.0 protocol latencies** are CXL 2.0 silicon figures assumed to carry
  over; no CXL 3.0 silicon exists to check them against.
* **HBM IDD currents** are calibrated to one published target; JEDEC leaves
  the values to vendor NDAs.
* **Routing logs** come from batch ≤ 64 on the profiling GPU.

## Deliberately unmodelled (harness properties, not device physics)

* **Replay quantum sampling.** `ratio = 0.9556 + 0.666 × (quantum/t_barrier)`,
  R² = 0.99997 on a controlled probe. Changing only `--quantum-ns` moved a
  ratio 1.023 → 1.228. Fitting the model to this would improve validation
  numbers while making the physics worse. It is no longer subtracted from the
  headline figures either (see A).

## Mistakes made and corrected during the campaign

Recorded because each was found by data rather than inspection, and the same
failure mode recurred:

1. **`bwRead::total` dilution.** A 1 µs idle tail inside the measurement
   window put every HBM bandwidth figure 4.8% low (707.8 → 736.6,
   1027.3 → 1072.1). The check band had been set from the same flawed
   measurement, so it passed the flaw.
2. **Link-stage ceilings.** PCIe4 measured its own crossbar (23.3 GB/s cap);
   CXL3 ran at a 128 GB/s stage — *above* the FLIT-corrected nominal.
3. **CXL3 constant fitted twice** — 117.0 under the serial recurrence, then
   114.5 under gating over 15 points (spread 5.8 → 1.1 GB/s). Quote 114.5.
4. **`t0` folded into one branch of a `max()`**, silently dropped on
   memory-bound barriers, producing `total_s` ~8% below the compute floor.
   The t0-free validation convention was blind to it by construction;
   `check_recurrence_invariants()` now covers that blind spot.
5. **HBM3E state coefficients labelled "measured" when they were scaled**
   from HBM3 by the `bg_w` ratio — wrong by −6% / −6% / **+17%** on refresh.
   Now measured from an HBM3e run's own per-state cycle counters.
6. **Energy buckets grouped by DRAMsim3's split, not the model's.** Refresh
   counted as dynamic while the model bills it as background: totals right,
   every per-component ratio meaningless. The old 0.842–0.860 "composite"
   was two large opposing errors partly cancelling.
7. **A `--hbm-channels` scan cited as evidence.** Its invariance is a
   tautology — the per-channel share does not depend on channel count.
