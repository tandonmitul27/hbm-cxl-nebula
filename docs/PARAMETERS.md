# Parameters, and where they come from

<!-- @tandonmitul27 -- authored document. -->

Every number the model rests on, with its source and confidence. `make params`
prints this table live from `sim/system_params.py`, which is the machine-readable
version of this page and the single place any of it should be edited.

How these values were arrived at — and what had to be corrected along the way —
is in [CALIBRATION.md](CALIBRATION.md).

---

## HBM device models

| Parameter | Value | Source |
|---|---|---|
| Channels × width | 16 × 64 bit | JESD238A (JEDEC HBM3 standard) |
| Banks | 2 pseudo-ch × 4 BG × 4 banks | JESD238A §3.1.2 |
| Data rate | 6.4 Gbps/pin (HBM3) | JESD238A Table 92 top bin |
| Peak, derived | 819.2 GB/s per stack | 64 b × 6.4 Gbps × 16 ÷ 8 |
| **Measured** | **736.6 GB/s per stack** | this repo, `make bw-stack` |
| Array timings | Ramulator2 JESD238-cited preset | vendor-representative¹ |
| tFAW / tRRD | halved vs channel level | JESD238A Table 3 scopes them **per pseudo-channel** |
| Voltages | VDD 1.1 V | JESD238A Table 70 |
| HBM3E rate | 9.6 Gbps/pin → 1072.1 GB/s measured | vendor material; timings held from HBM3² |

¹ JESD238A deliberately does not fix array timing *values* — it says outright
that tRAS is "the analog value from the vendor datasheet". A
vendor-representative preset is the strongest source short of an NDA
datasheet.
² Declared ESTIMATE: same array generation, PHY rate scaled, tRFC moved to
the 450 ns density tier.

---

## CXL protocol path

| Parameter | Value | Source |
|---|---|---|
| Host bridge latency | 50 ns | CXL-DMSim Table III |
| Host protocol | 12 ns | CXL-DMSim Table III (ASIC) |
| Device protocol | 15 ns ASIC / 60 ns FPGA | CXL-DMSim Table III |
| Device FIFO | 48 entries | CXL-DMSim Table III |
| Round trip, by construction | 2 × 77 = 154 ns | the three above |
| **Measured added latency** | **159.7 ns** ASIC, **248.5 ns** FPGA | this repo |
| Silicon reference | 154 ns ASIC, 245 ns FPGA | real CXL devices, CXL-DMSim Table II |

Agreement is 4.7 % (ASIC) and 1.4 % (FPGA). Only the *delta* is comparable:
the published 130 ns local-DDR figure is full load-to-use including the
LLC-miss path, while ours is memory-port latency.

---

## Links

| Link | Nominal | Effective (measured) | Note |
|---|---|---|---|
| PCIe 4.0 x16 | 26 GB/s | 25.3 (97 %) | FloE comparison point |
| CXL 2.0 x16 | 63 GB/s | 50.7 (80 %) | PCIe 5.0 electricals; headline config |
| CXL 3.0 x16 | 121 GB/s | 106.6 (88 %) | PCIe 6.0; **requires 128-entry FIFOs** |

**Measure links at an exact-nominal stage.** The link is a single-port XBar
of `width = round(rate / clock)` bytes, and a 64 B packet occupies
`ceil(64 / width)` WHOLE cycles — so unless `width` divides 64 exactly, the
harness measures itself rather than the device. Use `--bus-ghz 6.5` for
PCIe4 (width 4) and `--bus-ghz 15.125` for CXL3 (width 8). At the stock
4 GHz, PCIe4 caps at 23.3 GB/s and CXL3 at 128 GB/s — the latter *above*
the FLIT-corrected nominal, which lets traffic appear faster than the link
can physically run. CXL2 is unaffected (width 16 divides 64).

CXL 3.0 needs deeper device buffering than any validated silicon: outstanding
requests scale as bandwidth × round-trip, so 121 GB/s × ~280 ns ÷ 64 B ≈ 530
in flight. With the silicon-validated 48-entry FIFOs the link caps at
~101 GB/s (84 %) for large sequential expert fetches, and ~86 GB/s (71 %)
for a sustained mix — the cap is `8 x entries x 64 B / RTT`, and a shallower
window queues less, so its RTT is ~242 ns rather than the deep-FIFO ~280 ns.
Quote CXL 3.0 results with both the buffering assumption and the traffic
pattern stated.

---

## Expander backend

| Device | Link | Backend | Power |
|---|---|---|---|
| Marvell Structera X 2504 | CXL 2.0 x16 | 4 ch DDR5-6400, 200 GB/s | < 30 W typical |
| Astera Labs Leo (all SKUs) | CXL 1.1/2.0 x16 | 2 ch DDR5-5600, 2 TB | — |

Modelled as **8 × DDR5-6400 32-bit subchannels = 204.8 GB/s**, matching the
Structera part. Both vendors provision backend bandwidth *above* the x16 link,
which is why "the link binds" is an observation about real devices rather than
an assumption.

---

## Energy

| Term | Value | Source |
|---|---|---|
| HBM3 read | 3.261 pJ/bit | calibrated, see below |
| HBM3 total, sequential | **3.97 pJ/bit** | matches the anchor by construction |
| HBM3E total | 3.97 pJ/bit | held equal to HBM3 (conservative) |
| DDR5 media | **11.8 pJ/bit** | Micron IDD via gem5, **uncalibrated** |
| Background power | 0.252 W/HBM channel, 0.64 W/DDR5 subch | measured refresh + standby |
| CXL link + controller | **2 – 19 pJ/bit, swept** | bracketed, see below |

**Anchor** — O'Connor et al., *Fine-Grained DRAM*, MICRO'17: HBM access energy
**3.97 pJ/bit** (activation 1.21 + on-die movement 2.24 + interposer I/O 0.3,
including ECC).

**Why HBM energy is calibrated.** JESD238A fixes the voltages, the geometry
and the IDD measurement *states*, but deliberately leaves the IDD current
*values* to vendor datasheets — its value table is an empty template. Nobody
can quote HBM3 currents "from JEDEC". The `[power]` sections therefore keep the
HBM2 template's state *ratios* and scale the whole family by **one factor**
(HBM3 k = 3.270, HBM3E k = 4.648) until measured energy equals the anchor. One
target, one knob, labelled as such in the config files. HBM3E is held equal per
bit, so no generation-relative HBM energy claims are possible.

Sanity: 1.40 W/channel × 16 ≈ 22 W/stack → ~112 W for five stacks, against
3.9 pJ/bit × 3.35 TB/s = 104 W.

**Why DDR5 energy is not calibrated.** Nothing there was tuned. That it lands
at 11.8 pJ/bit — inside the DDR/GDDR device-class range — is a check, not a fit.

**Why link energy is a bracket.** No DRAM simulator models a serial link, and
no vendor publishes the SerDes-versus-controller split. The bracket is a
published PCIe 5.0 long-reach PHY at **11.4 pJ/bit** (CICC 2020), short-reach
on-package links near 0.5, and a hard whole-device ceiling of **18.75 pJ/bit**
(30 W ÷ 200 GB/s from the Structera brief). Any CXL access-energy figure must
add this term explicitly and report across the range.

Measured energy is reported by `sim/measure_energy.py`, which applies a unit
conversion DRAMsim3 omits — see [units](CALIBRATION.md#units).

---

## GPU anchors

| GPU | Capacity | Bandwidth | Stacks | HBM |
|---|---|---|---|---|
| H100 SXM | 80 GB | 3350 GB/s | 5 | HBM3 |
| H200 SXM | 141 GB | 4800 GB/s | 6 | HBM3E |
| B200 | 192 GB | 8000 GB/s | 8 | HBM3E |

---

## Models in the static map

| Model | Experts | top-k | MoE layers | Expert size (fp16) |
|---|---|---|---|---|
| OLMoE-1B-7B | 64 | 8 | 16 | 12.0 MiB |
| DeepSeek-V2-Lite | 64 | 6 | 26 | 16.5 MiB |
| Phi-3.5-MoE | 16 | 2 | 32 | 150.0 MiB |
| Mixtral-8x7B | 8 | 2 | 32 | 336.0 MiB |

Shapes come from each checkpoint's published config; `mapping/geometry/`
carries model shapes only. How they are placed across the tiers is in
[ARCHITECTURE.md](ARCHITECTURE.md#where-the-weights-go).
