"""Static address map: model weights -> physical addresses, HBM vs CXL.

===========================================================================
 @tandonmitul27  --  AUTHORED FILE (new)
===========================================================================

WHY THIS FILE EXISTS
    The memory harnesses in sim/ characterise the two tiers; this module
    is what decides WHICH BYTES GO WHERE.  It turns a model's shape
    (mapping/geometry/*.json) into a concrete static placement: a fixed
    address for every weight region, and a fixed answer to "is this
    expert in HBM or in CXL?".  Without it the tiers are characterised
    but nothing is placed in them.

THE DESIGN DECISION THAT MATTERS
    HBM holds a CACHE OVER CXL-resident experts, not an exclusive split.
    Every routed expert has a permanent canonical home in CXL; HBM holds
    (a) regions every token touches, pinned forever, and (b) copies of a
    chosen subset of experts.
    Under the static mapping shipped here, that subset is fixed at
    configuration time and never changes during execution -- which is
    exactly what makes the placement STATIC.  The layout would also
    admit a policy that changed the subset at run time, but no such
    policy is part of this repository.

WHY A CACHE RATHER THAN A SPLIT
    An exclusive flat-NUMA-style split cannot express "this expert is in
    both places", so it cannot express a preloaded working set at all,
    and it would force the CXL side to be re-laid-out whenever the HBM
    budget changed.  With a cache the CXL layout is invariant and only
    the resident set moves.

TWO CONSEQUENCES THAT SHAPE EVERYTHING DOWNSTREAM
    * Expert weights are READ-ONLY, so eviction costs nothing: no
      write-back, no dirty tracking, no coherence traffic.
    * Attention, shared-expert and embedding weights are NOT tiered.
      Every token touches them at every layer, so placing them in CXL
      would add traffic no placement could avoid.  They set a hard floor
      on HBM capacity.

ALIGNMENT -- read this before changing ALIGN
    2 MiB alignment applies to base ADDRESSES only, never to sizes.
    Rounding a size up would overstate the bytes actually fetched:
    DeepSeek's 16.5 MiB expert would bill as 18 MiB and inflate every
    fetch time by ~4%.

USAGE
    python mapping/address_map.py --tag Mixtral-8x7B --hbm-gib 80
    python mapping/address_map.py --tag OLMoE-1B-7B --hbm-gib 24 --dump
===========================================================================

Layout
------

    0x0000_0000  +--------------------------------+
                 |  HBM: always-resident           |   attention, shared
                 |       (attention, shared        |   experts, dense FFN,
                 |        experts, dense FFN,      |   embeddings.  Never
                 |        embeddings)              |   tiered -- every token
                 |                                 |   touches these.
     hbm_pinned  +--------------------------------+
                 |  HBM: expert cache              |   capacity is the
                 |       (policy-managed)          |   experiment's main axis
        hbm_end  +--------------------------------+
                 |  CXL: canonical home of every   |   full copy, always
                 |       routed expert             |   present
                 +--------------------------------+

**Every routed expert has a CXL home address.**  HBM holds a *cache* over that
home rather than an exclusive placement.  One mechanism then expresses the
whole O5 policy ladder: static pinning is a cache preloaded and locked, LRU is
the same cache with eviction, prefetch is the same cache filled early.  An
exclusive flat-NUMA split cannot express caching, so it would have forced a
second address map.

Consequences that matter downstream:

* Expert regions are contiguous and aligned, so fetching one is a single
  sequential burst -- which is what `createLinear` models in gem5 and what the
  bandwidth calibration measured.
* Attention and shared-expert weights are *not* tiered.  Every token uses them
  at every layer, so placing them in CXL would add traffic no policy can
  avoid.  They set a hard floor on HBM capacity.

    python address_map.py --tag DeepSeek-V2-Lite --hbm-gib 24
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "geometry"      # model shapes only; no routing data

# 2 MiB alignment applies to base ADDRESSES only, never to sizes.  Rounding a
# size up would overstate the bytes actually fetched -- DeepSeek's 17.3 MiB
# expert would bill as 18 MiB, inflating every fetch time by 4%.
ALIGN = 2 * 1024 * 1024
GIB = 1024 ** 3


def _align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------
# parameter counts
# --------------------------------------------------------------------------

def attention_params(att: dict) -> int:
    """Weights of one attention block.

    Standard MHA/GQA: Q and O are h x (n_heads*head_dim); K and V are
    h x (n_kv_heads*head_dim).  DeepSeek's MLA factors these through a latent
    projection instead, so it is counted separately.
    """
    h = att["hidden"]
    if att["mla"]:
        qk = att["qk_nope"] + att["qk_rope"]
        q = (att["q_lora"] * (h + att["n_heads"] * qk) if att["q_lora"]
             else h * att["n_heads"] * qk)
        kv_a = h * (att["kv_lora"] + att["qk_rope"])
        kv_b = att["kv_lora"] * att["n_heads"] * (att["qk_nope"] + att["v_head"])
        o = att["n_heads"] * att["v_head"] * h
        return q + kv_a + kv_b + o
    hd, nh, nkv = att["head_dim"], att["n_heads"], att["n_kv_heads"]
    return 2 * h * nh * hd + 2 * h * nkv * hd      # Q,O + K,V


def embedding_params(att: dict) -> int:
    """Token embedding plus lm_head, unless tied."""
    n = att["vocab"] * att["hidden"]
    return n if att["tie"] else 2 * n


# --------------------------------------------------------------------------

@dataclass
class Region:
    name: str
    tier: str            # "HBM" | "CXL"
    kind: str            # attention | shared | dense_ffn | embedding | expert
    base: int
    size: int
    layer: int | None = None
    expert: int | None = None


class AddressMap:
    def __init__(self, tag: str, dtype_bytes: int = 2, hbm_gib: float = 24.0):
        self.tag = tag
        self.dtype = dtype_bytes
        meta = json.loads((DATA / f"{tag}.json").read_text())
        self.g = meta["geometry"]
        shapes = json.loads((DATA / "_attention_shapes.json").read_text())
        self.att = shapes[tag]

        g = self.g
        # Exact sizes (what gets read); strides are the aligned allocation step.
        self.expert_bytes = g["expert_params"] * dtype_bytes
        self.expert_stride = _align_up(self.expert_bytes)
        self.att_bytes = attention_params(self.att) * dtype_bytes
        self.shared_bytes = g["shared_params_per_layer"] * dtype_bytes
        # A dense layer's FFN is the same shape as an expert but full width;
        # geometry stores it as expert_ff when the model is the dense contrast.
        self.dense_ffn_bytes = self.expert_bytes if g["dense_layers"] else 0
        self.emb_bytes = embedding_params(self.att) * dtype_bytes

        self.regions: list[Region] = []
        self._build(hbm_gib)

    # -- layout ------------------------------------------------------------

    def _build(self, hbm_gib: float):
        g = self.g
        cur = 0

        def add(name, tier, kind, size, layer=None, expert=None):
            nonlocal cur
            self.regions.append(Region(name, tier, kind, cur, size, layer, expert))
            cur += _align_up(size)          # stride aligned, size exact

        # ---- HBM, always resident ----------------------------------------
        add("embeddings", "HBM", "embedding", self.emb_bytes)
        for L in range(g["num_layers"]):
            add(f"attn.{L}", "HBM", "attention", self.att_bytes, layer=L)
        for L in g["dense_layers"]:
            add(f"dense_ffn.{L}", "HBM", "dense_ffn", self.dense_ffn_bytes, layer=L)
        if g["num_shared_experts"]:
            for L in g["moe_layers"]:
                add(f"shared.{L}", "HBM", "shared", self.shared_bytes, layer=L)

        self.hbm_pinned_end = cur
        self.pinned_bytes = cur

        # ---- HBM, policy-managed expert cache -----------------------------
        cache = int(hbm_gib * GIB) - self.pinned_bytes
        self.cache_bytes = max(0, cache)
        self.cache_slots = self.cache_bytes // self.expert_stride
        cur += self.cache_bytes
        self.hbm_end = cur

        # ---- CXL: canonical home of every routed expert -------------------
        self.cxl_base = cur
        for L in g["moe_layers"]:
            for e in range(g["num_experts"]):
                add(f"expert.{L}.{e}", "CXL", "expert", self.expert_bytes,
                    layer=L, expert=e)
        self.cxl_end = cur

    # -- the lookup Phase 2 needs ------------------------------------------

    def expert_home(self, layer: int, expert: int) -> tuple[int, int]:
        """CXL home address of a routed expert. O(1), no table walk."""
        try:
            i = self.g["moe_layers"].index(layer)
        except ValueError:
            raise KeyError(f"layer {layer} is not an MoE layer in {self.tag}")
        if not 0 <= expert < self.g["num_experts"]:
            raise KeyError(f"expert {expert} out of range")
        n = i * self.g["num_experts"] + expert
        return self.cxl_base + n * self.expert_stride, self.expert_bytes

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        g = self.g
        n_experts = g["num_experts"] * len(g["moe_layers"])
        routed_total = self.expert_bytes * n_experts          # exact bytes
        routed_span = self.expert_stride * n_experts          # address span
        return {
            "tag": self.tag,
            "dtype_bytes": self.dtype,
            "expert_bytes": self.expert_bytes,
            "expert_stride": self.expert_stride,
            "cxl_routed_span": routed_span,
            "attention_bytes_per_layer": self.att_bytes,
            "shared_bytes_per_layer": self.shared_bytes,
            "embedding_bytes": self.emb_bytes,
            "hbm_pinned_bytes": self.pinned_bytes,
            "hbm_cache_bytes": self.cache_bytes,
            "hbm_cache_slots": self.cache_slots,
            "cxl_routed_bytes": routed_total,
            "total_model_bytes": self.pinned_bytes + routed_total,
            "experts_total": n_experts,
            # capped at 1.0: a cache with more slots than there are experts
            # holds all of them, it does not hold them more than once.
            # (@tandonmitul27 -- the uncapped ratio is kept alongside so a
            # caller can still see how much HBM headroom is left over.)
            "resident_fraction": min(1.0, self.cache_slots / max(1, n_experts)),
            "cache_slots_per_expert": self.cache_slots / max(1, n_experts),
            "cxl_base": self.cxl_base,
            "cxl_end": self.cxl_end,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hbm-gib", type=float, default=24.0)
    ap.add_argument("--dtype-bytes", type=int, default=2, help="2=fp16, 1=fp8")
    ap.add_argument("--dump", action="store_true", help="write the region list")
    a = ap.parse_args()

    m = AddressMap(a.tag, a.dtype_bytes, a.hbm_gib)
    s = m.summary()
    mib = lambda b: b / 2**20
    print(f"=== {a.tag}  (fp{8*a.dtype_bytes}, HBM {a.hbm_gib} GiB) ===")
    print(f"  expert                {mib(s['expert_bytes']):>10.1f} MiB")
    print(f"  attention / layer     {mib(s['attention_bytes_per_layer']):>10.1f} MiB")
    if s["shared_bytes_per_layer"]:
        print(f"  shared experts /layer {mib(s['shared_bytes_per_layer']):>10.1f} MiB")
    print(f"  embeddings            {mib(s['embedding_bytes']):>10.1f} MiB")
    print(f"  --")
    print(f"  HBM pinned (untierable) {s['hbm_pinned_bytes']/GIB:>8.2f} GiB")
    print(f"  HBM expert cache        {s['hbm_cache_bytes']/GIB:>8.2f} GiB "
          f"= {s['hbm_cache_slots']} experts")
    print(f"  CXL routed experts      {s['cxl_routed_bytes']/GIB:>8.2f} GiB "
          f"= {s['experts_total']} experts")
    print(f"  total model             {s['total_model_bytes']/GIB:>8.2f} GiB")
    fits = s["cache_slots_per_expert"] >= 1.0
    print(f"  resident fraction       {s['resident_fraction']*100:>8.1f}%"
          + ("   (whole model fits; "
             f"{(s['cache_slots_per_expert']-1)*100:.0f}% HBM headroom spare)"
             if fits else "   -> the rest are fetched from CXL"))

    out = HERE.parent / "out" / "addrmap"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}_hbm{a.hbm_gib:g}.json").write_text(json.dumps(
        {"summary": s,
         "regions": [asdict(r) for r in m.regions] if a.dump else "use --dump"},
        indent=2))


if __name__ == "__main__":
    main()
