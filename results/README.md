# Sweep results

Produced by `analytical/sweep.py` under the model and constants validated by the
gem5 replay campaign — 103 operating points, mean ratio 0.996, a single
prediction within ±2.4% (MODEL.md §5b, full record in
[../docs/VALIDATION_CAMPAIGN.md](../docs/VALIDATION_CAMPAIGN.md)).

Earlier generations are archived and **must not be quoted**: `v1-preclib/`
(pre-calibration), `v2-nogating/` (pre arrival-gating), `v3-lowbw/`
(pre HBM-bandwidth fix), `v4-h100only/` (H100 only, saturated background
power). Each carries a `PROVENANCE.md` saying what superseded it.

**The `.parquet` files are not in git** — they are generated. `make sweep`
rebuilds `sweep.parquet` in ~30 minutes. A committed copy would silently
outlive the next change to the model; the tables below are the record.

## Files

| file | rows | what |
|---|---|---|
| `sweep.parquet` | 80,640 | **GPU {H100, H200, B200}** × 4 MoE models × batch {1,4,16,64} × capacity {0,10,25,50,75,100}% + the GPU's native HBM anchor × links {24.9, 50.7, 114.5} × policies {none, static, lru, lru_layer, oracle} × prefetch depth {0,1,2,4} × {fp16, fp8} × {decode, prefill}. Energy/EDP at e_link = {2, 10, 19} pJ/bit per row. 26,880 rows per GPU. |
| `dense.parquet` | 6,720 | Qwen2.5-3B-dense — the no-sparsity **control**, kept separate deliberately: folding it into `sweep.parquet` silently pollutes any aggregate taken over "the models". |
| `dynamic.parquet` | 108 | policy comparison |
| `critical.parquet` | 192 | cross-profiled static vs dynamic |

## Headline: capacity, not bandwidth, is the cliff

Native HBM capacity, batch 16, decode fp16, CXL 2.0, static placement:

| GPU | HBM | model | resident | ms/step | compute floor | stall |
|---|---|---|---|---|---|---|
| H100 | 80 GiB | OLMoE-1B-7B | 100% | 3.7 | 3.7 | 0% |
| H100 | 80 GiB | DeepSeek-V2-Lite | 100% | 6.9 | 6.9 | 0% |
| H100 | 80 GiB | Phi-3.5-MoE | 100% | 18.6 | 18.6 | 0% |
| H100 | 80 GiB | **Mixtral-8x7B** | **91%** | **169.8** | 25.7 | **93%** |
| H200 | 141 GiB | OLMoE-1B-7B | 100% | 2.2 | 2.2 | 0% |
| H200 | 141 GiB | DeepSeek-V2-Lite | 100% | 4.2 | 4.2 | 0% |
| H200 | 141 GiB | Phi-3.5-MoE | 100% | 10.9 | 10.9 | 0% |
| H200 | 141 GiB | **Mixtral-8x7B** | **100%** | **15.0** | 15.0 | **0%** |
| B200 | 192 GiB | OLMoE-1B-7B | 100% | 1.8 | 1.8 | 0% |
| B200 | 192 GiB | DeepSeek-V2-Lite | 100% | 3.3 | 3.3 | 0% |
| B200 | 192 GiB | Phi-3.5-MoE | 100% | 8.3 | 8.3 | 0% |
| B200 | 192 GiB | Mixtral-8x7B | 100% | 11.4 | 11.4 | 0% |

**The last 9% of residency costs 11x.** Mixtral at fp16 does not fit an
80 GiB H100 — 91% of its experts do — and that 9% miss rate is worth 144 ms
of stall per step, 93% of the step. Move the same model to a 141 GiB H200 and
it fits: 15.0 ms/step, stall-free — an **11.3x** speedup where the memory
system got only 1.75x wider. The capacity and the bandwidth grew together
(1.76x and 1.75x), and the capacity is what paid. Quantising to fp8 does the
same thing on the H100 itself: 13.2 ms/step at 100% residency, a 12.9x win
that costs nothing but precision.

That is the shape of the result: for a tiered memory system, the question is
whether the working set fits, not how fast the tier below is. The link only
matters once it doesn't.

## Where a link *does* matter

Once a model must tier, link bandwidth scales the stall almost linearly
(Mixtral b16 on 80 GiB H100, lru_layer):

| link | effective GB/s | ms/step |
|---|---|---|
| PCIe 4.0 x16 | 24.9 | 552.4 |
| CXL 2.0 x16 | 50.7 | 275.2 |
| CXL 3.0 x16 | 114.5 | 126.1 |

And the capacity curve is steep (H100, lru_layer, b16, CXL2, vs the
100%-resident static baseline):

| residency | slowdown |
|---|---|
| 0% | 58.6x – 70.2x |
| 25% | 43.0x – 53.3x |
| 50% | 27.6x – 36.2x |
| 75% | 12.1x – 19.1x |
| 100% | 1.5x – 3.6x |

## Reading these correctly

**`total_s` is achieved time; `compute_s` is the stall-free floor.** They are
equal only when nothing misses — which is why every 100%-resident row shows
identical achieved and floor columns, and Mixtral on H100 does not.
`check.py` asserts `total_s >= compute_s`; a bug during development produced
totals ~8% below the floor. **Quote `total_s`** — the v1 README quoted
Mixtral's *compute floor* (26.2 ms) as if it were achieved time.

**Residency is not hit rate.** A model that fits entirely means static *is*
the compute floor, and a dynamic policy can only lose ground by evicting
something it needed — hence lru_layer being 1.5–3.6x worse at full residency.
That is a property of the policy, not of the tier.

**Policy differences are largest where capacity is tightest.** Prefetch depth
≥ 1 is neutral-to-worse in stall-dominated regimes: a serialized link cannot
be conjured into more bandwidth, and staging slots cost hits.

**Energy conclusions must be reported across the `e_link` bracket** (2–19
pJ/bit): energy ×1.13 median and ×1.40 at the extreme, EDP likewise.
Orderings are preserved throughout.

**All three GPUs are replay-validated** against gem5 — no row here is
extrapolation along the GPU axis. HBM3e systems (H200, B200) carry a measured
**+1.5% bias** relative to H100, documented and unexplained in
[../docs/VALIDATION_CAMPAIGN.md](../docs/VALIDATION_CAMPAIGN.md). It sits
inside the ±2.4% prediction interval and reorders nothing.

## Model and constants behind these numbers

* **Per-expert arrival gating** — resident experts' reads and GEMVs overlap an
  in-flight fetch; only the missed expert's read waits. Improves every regime
  with no fitted parameter.
* **State-resolved background power** — `P_bg(u) = P_ref + u·P_act_stb +
  (1−u)·P_pre_stb`, with duty cycles reported by the recurrence. Charging the
  saturated endpoint over the whole window over-billed background by ~13% in
  exactly the stall-dominated regime these results occupy.
* **Per-barrier host overhead `t0`** — bracketed 5–40 µs, default 20, applied
  serially before both the memory and compute paths.
* **Prefill attention-score quadratic**, and a Little's-law device-buffering
  cap `min(rate, 8 × entries × 64 B / RTT)`.
* **HBM3 736.6 / HBM3e 1072.1 GB/s/stack** — corrected from 707.8 / 1027.3,
  which were diluted 4.8% by an idle tail inside the calibration's
  measurement window.
* **Links 24.9 / 50.7 / 114.5 GB/s** — PCIe4 and CXL3 re-measured at
  exact-nominal link stages after their originals proved to be
  harness-ceiling artifacts. CXL2 unchanged and never affected.

## Regenerate

```
python analytical/sweep.py                 # ~30 min, writes sweep.parquet
python docs/confidence_analysis.py     # accuracy / sensitivity / coverage
```
