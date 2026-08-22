> ## ⚠️ STALE — superseded by [MODEL.md](MODEL.md)
>
> This document is kept for its derivations and its framing of the
> optimisation levers. **Its constants are pre-calibration and must not be
> quoted.** Every hardware value in §1.1 was later corrected by the gem5
> replay campaign, and the background-power term was replaced outright:
>
> | here | corrected | why |
> |---|---|---|
> | `B_hbm1` 707.8 GB/s | **736.6** | a 1 µs idle tail diluted the measurement window by 4.8% |
> | HBM3E 1027.3 GB/s | **1072.1** | same dilution |
> | `B_HBM` 3539 GB/s, `α` 69.8 | **3683**, `α` 72.6 | follows from the above |
> | PCIe4 23.5 GB/s | **24.9** | the link stage was measuring its own crossbar ceiling |
> | CXL3 86 / 105.6 GB/s | **114.5** | stage overshot the FLIT-corrected nominal; refitted for fetch traffic |
> | `P_hbm_ch` 0.252 W/ch flat | **state-resolved** | `P_ref + u·P_act_stb + (1−u)·P_pre_stb`; the flat value over-charged stall-dominated runs by ~13% |
>
> The model described here is also now *implemented* (`analytical/`) and validated
> against gem5 at 103 operating points. For the equations as built, their
> provenance and their measured accuracy, read
> **[MODEL.md](MODEL.md)** and **[VALIDATION_CAMPAIGN.md](VALIDATION_CAMPAIGN.md)**.

# Analytical model for the HBM + CXL two-tier MoE memory system

Companion to `tandonmitul27/hbm-cxl-nebula`. This document defines the closed-form
time and energy model that replaces per-run gem5 simulation, plus the optimisation
levers that follow from the equations.

**Division of labour.** gem5 + DRAMsim3 supply *hardware constants* (achievable
bandwidth, added latency, pJ/bit). They never ran an MoE workload. This model
supplies everything above that line: routing, hit rates, batching, overlap,
per-token cost. The simulator is the calibration source; this is the evaluation
engine.

---

## 1. Parameters

### 1.1 Hardware (measured — do not re-derive from spec sheets)

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| `B_hbm1` | HBM3 bandwidth, one stack | 707.8 GB/s | `make bw-stack` |
| `n_st` | stacks | 5 (H100) / 6 (H200) / 8 (B200) | GPU anchor |
| `B_HBM` | near-tier bandwidth | `n_st · B_hbm1` = 3539 GB/s | derived |
| `B_hbm1(3E)` | HBM3E per stack | 1027.3 GB/s | `make bw-stack` |
| `B_CXL` | far-tier effective bandwidth | 50.7 GB/s (CXL 2.0 x16) | `make cxl-bandwidth` |
| — | alternatives | 23.5 (PCIe4 x16), 86 (CXL3 @48-FIFO), 105.6 (CXL3 @128-FIFO) | same |
| `t_CXL` | CXL added latency | 159.7 ns (ASIC), 248.5 ns (FPGA) | `make cxl-latency` |
| `t_HBM` | HBM device latency | ≈ 34 ns row-miss (tRP+tRCD) | `make check` |
| `e_HBM` | HBM3/3E access energy | 3.97 pJ/bit | calibrated to O'Connor MICRO'17 |
| `e_DDR5` | expander media energy | 11.8 pJ/bit | uncalibrated, device-class check |
| `e_link` | CXL SerDes + controller | **swept 2 – 19 pJ/bit** | bracketed, no simulator support |
| `P_hbm_ch` | HBM background power | 0.252 W/channel, 16 ch/stack | measured refresh+standby |
| `P_ddr_sch` | DDR5 background power | 0.64 W/subchannel, 8 subch | measured |
| `η_rw` | mixed read/write derate | 0.95 | `make check` |

Derived aggregates for the H100 anchor:

```
B_HBM   = 5 × 707.8      = 3539 GB/s
P_bg    = 0.252×16×5 + 0.64×8 = 20.16 + 5.12 = 25.28 W
α       = B_HBM / B_CXL  = 3539 / 50.7 = 69.8      <- the bandwidth ratio
```

`α` is the single most important number in the model. Everything about the
architecture reduces to it.

### 1.2 Model geometry (from `mapping/geometry/`)

| Symbol | Meaning |
|---|---|
| `E` | experts per MoE layer |
| `k` | top-k routing |
| `L_moe` | number of MoE layers |
| `S_e` | bytes per expert |
| `S_attn` | attention weight bytes per layer |
| `S_emb` | embedding bytes |
| `S_sh` | shared-expert bytes per layer (0 for Mixtral/Phi) |

| Model | E | k | L_moe | S_e (fp16) |
|---|---|---|---|---|
| OLMoE-1B-7B | 64 | 8 | 16 | 12.0 MiB |
| DeepSeek-V2-Lite | 64 | 6 | 26 | 16.5 MiB |
| Phi-3.5-MoE | 16 | 2 | 32 | 150.0 MiB |
| Mixtral-8x7B | 8 | 2 | 32 | 336.0 MiB |

### 1.3 Configuration knobs

| Symbol | Meaning |
|---|---|
| `C_HBM` | HBM capacity budget (bytes) |
| `B` | batch size (tokens per decode step) |
| `q` | weight bytes per parameter (2 = fp16, 1 = int8, 0.5 = int4) |
| `s` | Zipf skew of expert popularity |
| `d` | prefetch lookahead depth, in layers |

---

## 2. Capacity model

```
C_pin   = S_emb + L_moe · (S_attn + S_sh)          # untierable
M       = floor( (C_HBM − C_pin) / S_e )           # total expert cache slots
m       = M / L_moe                                # slots per MoE layer
N_tot   = E · L_moe                                # total routed experts
f_res   = M / N_tot                                # resident fraction
```

Check against `make addrmap MODEL=Mixtral-8x7B HBM=80`:

```
C_pin = 500 MiB + 32×80 MiB = 3060 MiB = 2.99 GiB   ✓
M     = (80 − 2.99) GiB / 0.328125 GiB = 234        ✓
f_res = 234 / 256 = 91.4 %                          ✓
```

**`f_res` is a capacity fraction, not a hit rate.** Conflating the two is the
single easiest way to get this model wrong. Hit rate is §4.

Pinned regions set a hard floor on `C_HBM`: attention and shared experts are
touched by every token at every layer, so tiering them adds traffic no placement
can avoid. Embeddings are the arguable exception (touched once per token, not per
layer) — see lever L7.

---

## 3. Primitive cost equations

### 3.1 Transfer time

```
T_xfer(S, Bw, t_lat) = t_lat + S / Bw
```

Latency matters only below the crossover size:

```
S* = Bw · t_lat
S*_CXL = 50.7e9 × 159.7e-9 = 8.1 KB
```

At Mixtral's 336 MiB expert, `S_e / S* ≈ 43,000`. **The expert path is purely
bandwidth-bound.** Keep `t_lat` in the equations for completeness and to make the
small-model / KV-cache cases correct, but expect it to vanish for `S_e > 1 MiB`.

### 3.2 Hit and miss

A directory lookup decides the tier before the request is issued, so there is no
probe-and-retry term.

```
t_hit  = t_HBM + S_e / B_HBM

t_miss = ( t_CXL + S_e / B_CXL )     # fetch from CXL home
       + ( S_e / B_HBM )            # write into staging slot
       + ( t_HBM + S_e / B_HBM )    # read back for compute
```

The double HBM traversal on a miss is a **modelling choice inherited from the
repo's worked example**, not a physical necessity — a design that streams
CXL → compute directly would drop both terms. Make it a flag
(`--stream-on-miss`) rather than hard-coding it; it is worth ~3 % of miss time
but changes the energy accounting materially.

Bandwidth-bound limit:

```
t_miss / t_hit  →  α + 2  ≈  71.8
```

Mixtral / H100 numbers, reproducing ARCHITECTURE.md exactly:

```
t_hit  = 352.32e6 / 3539e9                        = 0.0996 ms
t_miss = 6.949 + 2(0.0996)                        = 7.148 ms
```

---

## 4. Hit rate

This is the part gem5 cannot give you and the reason the analytical model exists.

### 4.1 Coverage formulation

Let `π_i` be the probability that a single routing draw in a given layer selects
expert `i`, with `Σ_i π_i = 1`. Let `R` be the resident set (chosen offline,
frozen). Then for batch 1:

```
h = Σ_{i ∈ R} π_i
```

Hit rate is *coverage of the popularity mass*, not the fraction of experts held.

### 4.2 Zipf closed form

Rank experts by popularity, `π_(j) ∝ j^(−s)`. If the profile is perfect, `R` is
the top `m`:

```
h(m, s) = H(m, s) / H(E, s),     H(n, s) = Σ_{j=1}^{n} j^(−s)
```

This gives the returns-to-capacity curve directly. High `s` (skewed routing) →
`h` saturates after a few slots and small HBM budgets are nearly free. Low `s`
(uniform routing, which is what load-balancing losses are *designed* to produce)
→ `h ≈ m/E` and there is no cheap win. **Report every result as a function of
`s`**; it is the load-bearing assumption.

### 4.3 Profile mismatch

Static placement is exactly as good as its profile. Model runtime popularity
`π'` distinct from profile popularity `π`:

```
h_actual = Σ_{i ∈ R(π)} π'_i
```

The repo demonstrates the extreme: same layer, same hardware, `{2,5,7}` pinned
vs `{0,1,4}` pinned, 7.85 ms vs 57 ms. Parameterise the mismatch (e.g. rank
correlation, or fraction of top-m preserved) and sweep it. This is the
strongest argument in the project for dynamic placement.

### 4.4 Batching — distinct experts, not requests

An expert fetched once serves every token in the step that routed to it. With
`kB` independent draws per layer:

```
D    = Σ_{i=1}^{E} [ 1 − (1 − π_i)^(kB) ]        # distinct experts touched
D_hit  = Σ_{i ∈ R} [ 1 − (1 − π_i)^(kB) ]
D_miss = D − D_hit
```

Bounds: `D → kB` for small `B`, `D → E` for large `B`. Because `D` saturates at
`E` while work grows as `B`, **per-token miss cost falls roughly as 1/B once
`kB ≳ E`.** For Mixtral (`E`=8, `k`=2) saturation arrives by `B ≈ 8`; for OLMoE
(`E`=64, `k`=8) not until `B ≈ 30`. This asymmetry across models is a headline
result the batch-1 worked example cannot show.

---

## 5. Time model

### 5.1 Per MoE layer, per decode step

```
T_mem(layer) = D_hit · t_hit + D_miss · t_miss
             + (S_attn + S_sh) / B_HBM            # pinned weights
             + T_kv                                # KV cache traffic
```

```
T_kv = 2 · B · L_ctx · d_kv · q / B_HBM
```

(`L_ctx` = context length, `d_kv` = KV dim per layer. Include it: at long context
and large batch it competes with expert traffic for the same HBM bandwidth.)

### 5.2 Compute and the memory-bound regime

Arithmetic intensity of a decode expert GEMM:

```
I = (2 · B · P FLOP) / (q · P bytes) = 2B / q  FLOP/byte
```

Machine balance for H100 (494.7 TFLOP/s dense bf16 ÷ 3.35 TB/s):

```
I_bal ≈ 148 FLOP/byte   →  memory-bound while B ≲ 148 (fp16)
```

For CXL-resident experts the balance point is `494.7e12 / 50.7e9 ≈ 9760`
FLOP/byte — **a miss is memory-bound at any batch size that fits in memory.**
Misses can never be compute-hidden by their own work.

### 5.3 Overlap

```
T_layer = T_mem(layer) + T_cmp(layer)          # no prefetch (repo baseline)
T_layer = max( T_mem(layer), T_cmp(layer) )    # perfect prefetch
```

Realistically, prefetch depth is bounded by when the router decision becomes
available. With lookahead `d` layers, the hideable work is `Σ_{j=1}^{d}
T_cmp(layer+j)`:

```
T_layer = max( T_mem(layer),  Σ_{j=1}^{d} T_cmp(layer+j) )
```

Since `T_cmp` is small in the memory-bound regime, prefetch helps far less than
intuition suggests — quantify this rather than assuming it.

### 5.4 Aggregate

```
T_step  = Σ_{ℓ ∈ MoE} T_layer(ℓ) + Σ_{ℓ ∉ MoE} T_dense(ℓ)
T_token = T_step / B
Throughput = B / T_step   tokens/s
```

---

## 6. Energy model

### 6.1 Dynamic

```
E_hit  = 8 · S_e · e_HBM

E_miss = 8 · S_e · (e_DDR5 + e_link)     # far-tier read + link crossing
       + 8 · S_e · e_HBM                 # staging write
       + 8 · S_e · e_HBM                 # read back
```

`e_link` is a **bracket, not a value** (2 – 19 pJ/bit). Every energy result must
be reported as a range across that sweep, and the bracket must be stated.

### 6.2 Static

```
E_static = P_bg · T_total
```

### 6.3 Total, and the result that matters

```
E_total = Σ E_hit + Σ E_miss + P_bg · T_total
```

Mixtral / H100 per-expert, with `e_link` at the mid bracket (11.4 pJ/bit):

| | dynamic | static | total |
|---|---|---|---|
| hit | 11.2 mJ | 2.5 mJ | **13.7 mJ** |
| miss | 87.8 mJ | 180.7 mJ | **268.5 mJ** |

Two things fall out immediately, and both are worth stating as findings:

1. **Dynamically, a miss is only ~7.8× a hit** — because DDR5 + link is only
   ~6× HBM's pJ/bit, while bandwidth is 70× lower. On dynamic energy alone,
   CXL tiering looks cheap.
2. **Once static power is included, a miss is ~20× a hit**, and 67 % of miss
   energy is idle HBM burning through a 7 ms stall. The energy penalty of CXL is
   overwhelmingly a *consequence of the latency penalty*, not of the media.

This is a genuine analytical result the microbenchmarks cannot produce, and it
means latency optimisations and energy optimisations are the same optimisations.

---

## 7. Optimisation levers

Differentiate the aggregate. In the bandwidth-bound limit, per layer:

```
T ≈ D · S_e/B_HBM · [ 1 + (1−h)(α + 1) ]
```

so

```
∂T/∂h  = −D · (S_e/B_HBM) · (α + 1)
```

Every point of hit rate is worth `α+1 ≈ 71` HBM reads. Ranked levers:

| # | Lever | Effect on `T` | Notes |
|---|---|---|---|
| L1 | **Raise `h`** (better profile, dynamic policy, more slots) | linear, gain ×(α+1) | dominant. Bounded by Zipf skew `s` |
| L2 | **Quantise** `q`: fp16→int8→int4 | `S_e ∝ q`, so `T ∝ q` on *both* tiers, **and** `M ∝ 1/q` so `h` rises too | compound win; strongest single knob |
| L3 | **Raise `B_CXL`**: CXL 3.0, wider link | `α ∝ 1/B_CXL` | constrained — see below |
| L4 | **Batch** | miss cost per token → `1/B` once `kB ≳ E` | free; changes model ranking |
| L5 | **Prefetch depth `d`** | hides `min(T_mem, Σ T_cmp)` | weak in memory-bound regime — quantify |
| L6 | **Stream on miss** (drop double HBM traversal) | `−2/α` ≈ −2.9 % | small, but free |
| L7 | **Un-pin embeddings** | `C_pin ↓ 500 MiB` → `M ↑` → `h ↑` | costs one CXL read per token; check the trade |
| L8 | **Bigger HBM** (H200 141 GB, B200 192 GB) | `M ↑`, `B_HBM ↑` | changes both terms; the trivial baseline |

**Constraint on L3.** Link bandwidth is not free to scale — outstanding-request
capacity binds first:

```
N_outstanding ≥ B_CXL · RTT / 64 B
```

At CXL 3.0 x16: `121e9 × 280e-9 / 64 ≈ 530` requests, versus 8 bridges × 48
entries = 384 available in silicon-validated designs. That is why the repo
reports 86 GB/s (71 %) at 48-entry FIFOs and 105.6 GB/s (87 %) only at
128-entry. **Any CXL 3.0 result must carry the buffering assumption explicitly.**

---

## 8. Validation targets

The analytical model must reproduce the simulator-derived numbers before it is
trusted for anything new. Build these as unit tests:

| Target | Expected | Source |
|---|---|---|
| `t_hit` (Mixtral, H100) | 0.10 ms | ARCHITECTURE.md |
| CXL fetch of one expert | 6.95 ms | ARCHITECTURE.md |
| `t_miss` | 7.15 ms | derived from the 4-step trace |
| 4-step trace, 3 slots, `{2,5,7}` pinned | 7.85 ms, 7/8 hits | ARCHITECTURE.md |
| Same trace, `{0,1,4}` pinned | ≈ 57 ms | ARCHITECTURE.md |
| Mixtral expert over PCIe4 x16 | 15.0 ms | FloE, ~15 ms |
| Capacity accounting, 4 models | ≤ 0.3 % error | `make addrmap` |
| HBM energy at full bandwidth | ~112 W for 5 stacks | 3.9 pJ/bit × 3.35 TB/s = 104 W |

If the analytical model reproduces all eight, the parameter plumbing is correct
and any divergence in new configurations is a modelling difference, not a bug.

---

## 9. Implementation structure

Keep placement policy separate from the cost model, mirroring the repo's split
between `mapping/address_map.py` and the memory model.

```
params.py       # import constants from sim/system_params.py — never retype them
geometry.py     # model shapes; reuse mapping/geometry/
capacity.py     # §2: C_pin, M, m, f_res
routing.py      # §4: π distribution (Zipf or empirical trace), D(B), h
policy.py       # resident-set selection: static-profile, oracle, LRU, popularity-adaptive
timing.py       # §3, §5
energy.py       # §6, with e_link swept
sweep.py        # experiment driver: model × HBM budget × B × s × q × link gen
validate.py     # §8
```

Two design rules worth enforcing from the start:

- **Single source of truth for constants.** Import from `sim/system_params.py`
  rather than copying values. If Mitul recalibrates, your model tracks it.
- **Mark estimates so they cannot reach a result unnoticed** — the repo's
  `NEEDS_SOURCE` convention. Carry it for `e_link`, the CXL 3.0 FIFO assumption,
  HBM3E timings, and any routing distribution not taken from a real trace.

---

## 10. Caveats to carry into the write-up

- The GPU-as-CXL-host single-hop topology does not exist in shipping hardware;
  real GPUs reach CXL memory through a CPU and pay a second traversal. This is
  the **optimistic case for CXL** and results must say so.
- HBM energy matches its anchor *because it was calibrated to it* — that check
  confirms the calibration holds, it is not independent evidence.
- `e_link` has no simulator support and no vendor PHY/controller split. It is a
  swept bracket, permanently.
- The double-HBM-traversal miss path is a modelling choice; flag it, don't bury
  it.
