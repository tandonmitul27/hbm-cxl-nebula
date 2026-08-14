# Architecture

<!-- @tandonmitul27 -- authored document. -->

The topology, the address map that places MoE weights across the two tiers,
and a worked example of a single placement decision.

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

Both tiers are real simulated memory: HBM3/HBM3E and DDR5-6400 device
models run in DRAMsim3, behind a CXL protocol path whose latency is
calibrated against measurements from shipping CXL silicon. Every value on
that diagram is sourced in [PARAMETERS.md](PARAMETERS.md).

This is the forward-looking single-hop topology. Shipping hardware attaches
CXL to a CPU, so a GPU reaching CXL memory pays a second traversal; the
single-hop case modelled here is the optimistic one for CXL, and results
should say so.

---

## Where the weights go

`mapping/address_map.py` lays the model out across one flat address space:

```
  HBM │ pinned: embeddings, attention, shared experts    never tiered
      │ expert cache: a fixed set of expert copies       the capacity knob
──────┼──────────────────────────────────────────────────────────────────
  CXL │ canonical home of every routed expert            always present
```

HBM holds a **cache over** the CXL-resident experts rather than an exclusive
split, so the CXL layout never changes when the HBM budget does — only the
resident set moves. Under the static mapping here that set is fixed at
configuration time. Expert weights are read-only, so eviction is free: no
write-back, no dirty tracking, no coherence.

Pinned regions are the reason HBM capacity has a floor. Attention and shared
experts are touched by every token at every layer, so tiering them would add
traffic no placement can avoid.

```console
$ make addrmap MODEL=Mixtral-8x7B HBM=80
=== Mixtral-8x7B  (fp16, HBM 80.0 GiB) ===
  expert                     336.0 MiB
  attention / layer           80.0 MiB
  embeddings                 500.0 MiB
  --
  HBM pinned (untierable)     2.99 GiB
  HBM expert cache           77.01 GiB = 234 experts
  CXL routed experts         84.00 GiB = 256 experts
  total model                86.99 GiB
  resident fraction           91.4%
```

Expert regions are contiguous and aligned, so fetching one is a single
sequential burst — which is what the bandwidth calibration in
[CALIBRATION.md](CALIBRATION.md) measured. Alignment applies to base
addresses only, never to sizes: rounding a size up would overstate the bytes
actually fetched.

---

## A worked example

Take Mixtral — 8 experts per layer, top-2 routing, **336 MiB** per expert —
and follow **one layer** through four decode steps at batch 1. Suppose this
layer's share of the HBM cache is 3 slots.

Three things exist, and only the first two occupy memory:

```
CXL   experts 0..7 for this layer, permanent homes, never move   8 x 336 MiB
HBM   3 cache slots  <- set by the HBM budget
DIR   which experts are currently in HBM  <- 8 bits. Metadata, not memory.
```

The timing constants come from the two tiers, and you can reproduce both
(`make bw-stack`, `make cxl-bandwidth`):

| Operation on one 336 MiB expert | Cost |
|---|---|
| read it from HBM (3540 GB/s, 5 stacks) | **0.1 ms** |
| fetch it over CXL (50.7 GB/s effective) | **6.95 ms** |
| write it into an HBM slot | **0.1 ms** |

A hit is ~0.1 ms and a miss ~7.05 ms — **70x apart**. That ratio is what the
placement decision is trading against.

Per layer, the router names its top-2 and then:

```
for each named expert:
    DIR says resident?  --yes-->  read from HBM                      0.1 ms
                        --no --->  read from its CXL home            6.95 ms
                                   write into an HBM slot            0.1 ms
```

Note what does *not* happen: no read is ever issued to HBM, allowed to fail,
and then retried against CXL. The directory is consulted first and the request
goes to exactly one tier. At 336 MiB granularity with a few hundred objects,
that metadata is trivially small — a residency bitmap for all 256 of Mixtral's
experts is 32 bytes — so there is no reason to build a probe path.

**The run.** Profile offline, pin the three most popular experts for this
layer — say `{2, 5, 7}` — load them before inference, and freeze:

| Step | Router picks | Directory | What runs | Time |
|---|---|---|---|---|
| 0 | 2, 5 | both resident | 2 HBM reads | 0.2 ms |
| 1 | 5, **3** | 3 not resident | CXL fetch of 3, then compute | 7.25 ms |
| 2 | 2, 7 | both resident | 2 HBM reads | 0.2 ms |
| 3 | 5, 2 | both resident | 2 HBM reads | 0.2 ms |

**≈ 7.85 ms; 7 of 8 expert requests hit.**

Step 1 is the one to look at. Expert 3 is fetched, used, and **not retained** —
the resident set is fixed, so its landing slot is staging that gets overwritten.
If a later step needs expert 3 again, it pays the full 6.95 ms again. That is
what makes the placement static.

**Static placement is exactly as good as its profile.** Hand the same layer a
profile taken from different traffic — say it pinned `{0, 1, 4}` — and all
eight requests miss instead of one: **≈ 57 ms**, seven times slower on
identical work. The layout did not change and neither did the hardware; only
the choice of what to pin did. Profile quality is the whole game, which is why
`mapping/address_map.py` keeps placement separate from the memory model and
lets you re-run any configuration in seconds.
