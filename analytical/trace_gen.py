"""Phase 2: routing log + address map + policy -> timed memory schedule.

Reads a Phase 1 routing log, decides where each expert lives under a given
placement policy, and plays the decode timeline forward against a link with
finite bandwidth.  Produces both:

  * an analytic result (hit rate, bytes moved, stall time, effective step time)
    -- cheap enough to sweep the whole (batch, capacity, bandwidth, policy)
    grid in seconds
  * a gem5 segment script for selected points, so the analytic model can be
    checked against the calibrated memory system

The policy is applied HERE, never baked into the routing log, so trying a
different one is a re-run of this script rather than another GPU pass.

Timeline model
--------------

Decode has a hard per-layer barrier and weights are read-only, so the schedule
is an exact recurrence rather than something needing feedback:

    for each layer:
        misses    = experts needed but not resident
        issue     = when the prefetcher could have started (D layers earlier)
        arrive    = max(issue, link_free) + miss_bytes / link_bw
        layer_beg = max(now, arrive)          <- stall if data is late
        now       = layer_beg + T_layer

`link_free` serialises transfers, so prefetching further ahead cannot conjure
bandwidth that is not there -- which is the constraint the whole study turns
on.

    python trace_gen.py --tag Mixtral-8x7B --batch 4 --hbm-gib 24 \
        --policy lru --link-gbps 52
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# AddressMap is single-sourced in mapping/ -- the placement is shared with
# the gem5 harnesses and must not fork.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mapping"))
from address_map import AddressMap, GIB          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DECODE = 1


# --------------------------------------------------------------------------
# demand: which experts each (step, layer) needs, unioned across the batch
# --------------------------------------------------------------------------

def demand_sequence(tag: str, batch: int,
                    phase: int = DECODE) -> tuple[list, dict, dict]:
    """[(step, layer, frozenset(experts))...] in execution order.

    The UNION across the batch is what matters: a batched decode step must have
    every expert any sequence in the batch selected, so per-sequence sparsity is
    not what the memory system sees.  For prefill (phase=0) the union runs
    across every prompt position as well -- one barrier per layer, which is
    the harshest tiering case.
    """
    # Only the run index is needed here; model GEOMETRY lives in
    # mapping/geometry/ and is read through AddressMap, so there is exactly
    # one copy of it in the repository and no way for the two to drift.
    index = json.loads((DATA / "routing" / "index.json").read_text())
    run = next((r for r in index.get(tag, [])
                if r.get("ok") and r["batch"] == batch), None)
    if run is None:
        raise SystemExit(f"{tag}: no completed run at batch {batch}")
    log = DATA / "routing" / tag / run["file"]
    if not log.exists():
        raise SystemExit(
            f"missing routing log: {log}\n"
            f"The routing logs ship with the repository; a missing one means "
            f"an incomplete checkout or an edited index.json. "
            f"Run `make routing` to see which of the 20 are present.")
    t = pq.read_table(log,
                      columns=["phase", "step", "layer", "expert"],
                      filters=[("phase", "==", phase)])
    d = {c: t.column(c).to_numpy() for c in t.column_names}
    order = np.lexsort((d["layer"], d["step"]))
    step, layer, exp = d["step"][order], d["layer"][order], d["expert"][order]

    seq, cur, key = [], set(), None
    for s, L, e in zip(step, layer, exp):
        k = (int(s), int(L))
        if k != key:
            if key is not None:
                seq.append((key[0], key[1], frozenset(cur)))
            key, cur = k, set()
        cur.add(int(e))
    if key is not None:
        seq.append((key[0], key[1], frozenset(cur)))
    return seq, run


def kv_bytes_per_tok_layer(att: dict, dtype_bytes: int) -> int:
    """KV-cache bytes one token adds per transformer layer.

    GQA/MHA: K and V, n_kv_heads x head_dim each.  MLA: the compressed
    latent (kv_lora + qk_rope) -- the architecture's intent.  (transformers
    5.x caches MLA *decompressed*, ~9x larger; that is an implementation
    artifact of the profiling harness, not of the modelled system, and is
    documented in data/README.md.)
    """
    if att["mla"]:
        return (att["kv_lora"] + att["qk_rope"]) * dtype_bytes
    return 2 * att["n_kv_heads"] * att["head_dim"] * dtype_bytes


# --------------------------------------------------------------------------
# policies: which experts are resident, and how residency evolves
# --------------------------------------------------------------------------

class Policy:
    """Base: `hit` reports residency and records the access."""

    def __init__(self, slots: int):
        self.slots = slots

    def resident(self, key) -> bool:
        raise NotImplementedError

    def admit(self, keys, idx: int):
        raise NotImplementedError


class NoCache(Policy):
    """Floor: nothing resident, every expert fetched every time."""

    def resident(self, key): return False
    def admit(self, keys, idx): pass


class StaticPopular(Policy):
    """Profile offline, pin the top-N most popular, never change.

    The obvious policy, and the one Phase 1 suggests will disappoint: measured
    routing skew is close to uniform.
    """

    def __init__(self, slots, seq):
        super().__init__(slots)
        cnt = defaultdict(int)
        for _s, L, experts in seq:
            for e in experts:
                cnt[(L, e)] += 1
        top = sorted(cnt, key=cnt.get, reverse=True)[:slots]
        self.pinned = set(top)

    def resident(self, key): return key in self.pinned
    def admit(self, keys, idx): pass


class LRU(Policy):
    """Global recency cache. Eviction is free -- weights are never dirty.

    Warning: this is pathological for MoE decode.  Layers are visited in a
    strict cycle, so a global LRU always holds the most recently executed
    layers; by the time the next token returns to layer 0 its experts have long
    been evicted.  Expect ~0% hit rate whenever the cache is smaller than one
    full pass.  Kept as a baseline precisely to show that.
    """

    def __init__(self, slots):
        super().__init__(slots)
        self.od: OrderedDict = OrderedDict()

    def resident(self, key): return key in self.od

    def admit(self, keys, idx):
        for k in keys:
            if k in self.od:
                self.od.move_to_end(k)
            else:
                self.od[k] = True
                if len(self.od) > self.slots:
                    self.od.popitem(last=False)


class Belady(Policy):
    """Offline optimal: evict whatever is used furthest in the future.

    Not implementable online -- it is the ceiling that says how much any
    cleverness could possibly buy.
    """

    def __init__(self, slots, seq):
        super().__init__(slots)
        self.future = defaultdict(list)
        for i, (_s, L, experts) in enumerate(seq):
            for e in experts:
                self.future[(L, e)].append(i)
        self.cursor = defaultdict(int)
        self.cache: set = set()

    def _next_use(self, key, idx):
        uses = self.future[key]
        c = self.cursor[key]
        while c < len(uses) and uses[c] <= idx:
            c += 1
        self.cursor[key] = c
        return uses[c] if c < len(uses) else float("inf")

    def resident(self, key): return key in self.cache

    def admit(self, keys, idx):
        if self.slots <= 0:
            return
        for k in keys:
            if k in self.cache:
                continue
            if len(self.cache) >= self.slots:
                victim = max(self.cache, key=lambda x: self._next_use(x, idx))
                if self._next_use(victim, idx) <= self._next_use(k, idx):
                    continue                      # k is worse; do not admit
                self.cache.discard(victim)
            self.cache.add(k)


class SLRUPerLayer(Policy):
    """Segmented LRU, partitioned per layer: each layer's slice is split
    into a probation segment (new admissions) and a protected segment
    (promoted on re-reference).  Protects frequently-reused experts from
    being churned out by one-shot admissions -- the classic fix for pure
    LRU under mixed frequency/recency traffic.
    """

    def __init__(self, slots, n_layers):
        super().__init__(slots)
        per = slots // max(1, n_layers)
        self.prot_cap = per // 2
        self.prob_cap = per - self.prot_cap
        self.prot: dict[int, OrderedDict] = defaultdict(OrderedDict)
        self.prob: dict[int, OrderedDict] = defaultdict(OrderedDict)

    def resident(self, key):
        L, e = key
        return e in self.prot[L] or e in self.prob[L]

    def admit(self, keys, idx):
        if self.prob_cap <= 0:
            return
        for L, e in keys:
            prot, prob = self.prot[L], self.prob[L]
            if e in prot:
                prot.move_to_end(e)
            elif e in prob:                     # re-reference: promote
                del prob[e]
                prot[e] = True
                if len(prot) > self.prot_cap:   # demote LRU protected
                    old, _ = prot.popitem(last=False)
                    prob[old] = True
            else:                               # miss: probation MRU
                prob[e] = True
            while len(prob) > self.prob_cap:
                prob.popitem(last=False)


class WLFUPerLayer(Policy):
    """Windowed LFU per layer: admit on miss, evict the expert with the
    lowest reference count over a sliding window of recent steps.  Sees the
    mid-term frequency structure that pure recency (LRU) misses, while
    forgetting fast enough to track drift -- aimed at the oracle-vs-LRU gap
    at low batch.
    """

    def __init__(self, slots, n_layers, window_barriers):
        super().__init__(slots)
        self.per = slots // max(1, n_layers)
        self.window = max(1, window_barriers)
        self.res: dict[int, set] = defaultdict(set)
        self.hist: dict[int, list] = defaultdict(list)   # per-layer key lists
        self.cnt: dict[int, defaultdict] = defaultdict(
            lambda: defaultdict(int))

    def resident(self, key): return key[1] in self.res[key[0]]

    def admit(self, keys, idx):
        if self.per <= 0 or not keys:
            return
        L = keys[0][0]
        es = [e for _, e in keys]
        self.hist[L].append(es)
        cnt = self.cnt[L]
        for e in es:
            cnt[e] += 1
        if len(self.hist[L]) > self.window:
            for e in self.hist[L].pop(0):
                cnt[e] -= 1
        res = self.res[L]
        for e in es:
            if e in res:
                continue
            if len(res) < self.per:
                res.add(e)
            else:
                victim = min(res, key=lambda x: cnt[x])
                if cnt[victim] < cnt[e]:
                    res.discard(victim)
                    res.add(e)


class WindowedPopular(Policy):
    """Dynamic migration: re-pin to the top-`slots` experts of a sliding
    demand window every `epoch` barriers.  Evictions are free (weights are
    clean); promotions are queued and move over IDLE link time only (the
    simulate() migration hook), so migration never delays a demand fetch.

    Demand misses are read THROUGH (served from CXL without admission) --
    residency changes only by migration.  This isolates "proactive
    placement" from "reactive caching" (LRU admits on miss); comparing the
    two at equal capacity is the point of the experiment.
    """

    def __init__(self, slots, epoch_barriers, window_barriers,
                 admit_on_miss=False, initial=None):
        super().__init__(slots)
        self.epoch = max(1, epoch_barriers)
        self.window = max(1, window_barriers)
        self.admit_on_miss = admit_on_miss    # hybrid: reactive fill too
        self.hist: list[list] = []            # per-barrier key lists
        self.res: set = set(list(initial)[:slots]) if initial else set()
        self.queue: list = []                 # promotions awaiting link idle

    def resident(self, key): return key in self.res

    def admit(self, keys, idx):
        self.hist.append(keys)
        if len(self.hist) > self.window:
            self.hist.pop(0)
        if self.admit_on_miss:
            # hybrid: a fetched miss lands in a free slot (it crossed the
            # link anyway); the windowed epoch below handles eviction
            for k in keys:
                if k not in self.res and len(self.res) < self.slots:
                    self.res.add(k)
        if (idx + 1) % self.epoch:
            return
        cnt = defaultdict(int)
        for ks in self.hist:
            for k in ks:
                cnt[k] += 1
        target = set(sorted(cnt, key=cnt.get, reverse=True)[:self.slots])
        self.res &= target                    # evict (free)
        # drop stale queued promotions, keep still-wanted ones in order
        self.queue = [k for k in self.queue if k in target]
        queued = set(self.queue) | self.res
        self.queue.extend(k for k in target if k not in queued)

    # ---- migration hook, called by simulate() with an idle-byte budget ----
    def migrate(self, budget_bytes: float, expert_bytes: float) -> float:
        moved = 0.0
        while self.queue and moved + expert_bytes <= budget_bytes:
            k = self.queue.pop(0)
            if len(self.res) < self.slots:
                self.res.add(k)
                moved += expert_bytes
        return moved


class LRUPerLayer(Policy):
    """LRU partitioned per layer -- the fix for the cyclic-access pathology.

    Each MoE layer gets its own slice of the cache, so an expert competes only
    with the other experts of its own layer, which is the reuse the routing log
    actually shows.
    """

    def __init__(self, slots, n_layers):
        super().__init__(slots)
        # floor division, NOT max(1, ...): granting every layer a slot when
        # slots < n_layers would overcommit capacity by up to n_layers-1
        self.per = slots // max(1, n_layers)
        self.od: dict[int, OrderedDict] = defaultdict(OrderedDict)

    def resident(self, key): return key[1] in self.od[key[0]]

    def admit(self, keys, idx):
        if self.per <= 0:
            return
        for L, e in keys:
            od = self.od[L]
            if e in od:
                od.move_to_end(e)
            else:
                od[e] = True
                if len(od) > self.per:
                    od.popitem(last=False)


def make_policy(name, slots, seq, n_layers=1, epoch=None, window=None):
    return {"none": lambda: NoCache(slots),
            "static": lambda: StaticPopular(slots, seq),
            "lru": lambda: LRU(slots),
            "lru_layer": lambda: LRUPerLayer(slots, n_layers),
            "adaptive": lambda: WindowedPopular(
                slots, epoch or n_layers, window or 8 * n_layers),
            "hybrid": lambda: WindowedPopular(
                slots, epoch or n_layers, window or 8 * n_layers,
                admit_on_miss=True),
            "slru_layer": lambda: SLRUPerLayer(slots, n_layers),
            "wlfu_layer": lambda: WLFUPerLayer(
                slots, n_layers, window or 16 * n_layers),
            "oracle": lambda: Belady(slots, seq)}[name]()


# --------------------------------------------------------------------------
# the timeline
# --------------------------------------------------------------------------

def simulate(seq, policy, expert_bytes, link_bps, t_layer_fn, depth,
             base_layer_bytes=0.0, hbm_bps=float("inf"),
             kv_layer_fn=None, collect=None, xfer_lat_s=0.0,
             t_compute_fn=None, t0_s=0.0):
    """Play the decode timeline forward. Returns aggregate statistics.

    `t_compute_fn(n_experts, step) -> s` is the pure COMPUTE floor of a
    barrier (t0 + FLOPs/peak, no memory term). Supplying it enables
    per-expert arrival gating; without it the floor falls back to the full
    roofline time, which is the pre-gating behaviour.

    `t_layer_fn(n_experts, step) -> s` is the per-barrier roofline time (see
    gpu_model.py); it varies with the batch-union size (prefill unions
    approach every expert) and with the step (KV cache grows as the sequence
    lengthens).

    `kv_layer_fn(step) -> (read_bytes, write_bytes)` is the KV-cache traffic
    this barrier adds on the HBM side; None disables it.

    `base_layer_bytes` is the attention + shared-expert traffic a layer always
    reads from HBM; together with the routed experts it makes the HBM-read
    byte count that the energy model consumes.

    Fill accounting: a missed expert crosses the link AND is written into the
    HBM cache, so its arrival pays nbytes/link + nbytes/hbm_write.  At
    link=52 GB/s vs HBM=3.5 TB/s that is a ~1.5% correction -- small, but
    zero-cost to include and it keeps the recurrence honest.
    """
    now = 0.0
    link_free = 0.0
    layer_begin: list[float] = []
    fetched = stalled = 0.0
    hbm_read = hbm_write = compute_s = 0.0
    hbm_busy_s = cxl_busy_s = 0.0     # for state-resolved background power
    hits = misses = 0

    for idx, (_step, L, experts) in enumerate(seq):
        keys = [(L, e) for e in experts]
        miss = [k for k in keys if not policy.resident(k)]
        hits += len(keys) - len(miss)
        misses += len(miss)

        nbytes = len(miss) * expert_bytes
        fetched += nbytes
        # compute reads every needed expert out of HBM (misses land there
        # first), plus the layer's always-resident weights, plus KV traffic
        kv_rd, kv_wr = kv_layer_fn(_step) if kv_layer_fn else (0.0, 0.0)
        layer_hbm = base_layer_bytes + len(keys) * expert_bytes + kv_rd + kv_wr
        hbm_read += layer_hbm - kv_wr
        hbm_write += kv_wr
        if collect is not None:
            collect.append({"step": int(_step), "layer": int(L),
                            "n_experts": len(keys),
                            "hbm_bytes": float(layer_hbm),
                            "cxl_bytes": float(nbytes)})
        # Prefetch depth D: the transfer could have been started when the layer
        # D positions earlier began.  Depth 0 means demand-fetch.
        issue = layer_begin[idx - depth] if depth and idx >= depth else now
        # Migration (WindowedPopular): queued promotions consume ONLY the
        # idle link window before this barrier's demand transfer, so they
        # can never delay a demand fetch.  Migrated bytes still cost CXL
        # reads + HBM fills (accounted like fetches).
        if hasattr(policy, "migrate") and issue > link_free:
            moved = policy.migrate((issue - link_free) * link_bps,
                                   expert_bytes)
            if moved:
                link_free += moved / link_bps
                fetched += moved
        start = max(issue, link_free)
        # xfer_lat_s: one loaded round-trip (link protocol + expander media
        # first access) per barrier fetch.  The stream pipelines behind the
        # first line, so latency is paid once per transfer, not per packet --
        # the same structure the gem5 replay exhibits.  At 12-336 MiB
        # transfers this is <0.01% of the fetch; it is carried so the model
        # DEMONSTRATES latency-insensitivity rather than assuming it, and
        # stays honest if run at finer granularity.
        arrive = start + (xfer_lat_s + nbytes / link_bps + nbytes / hbm_bps
                          if nbytes else 0.0)
        link_free = arrive

        # ---- per-expert arrival gating -------------------------------
        # A layer's RESIDENT experts are already in HBM, so their weight
        # reads and GEMVs proceed WHILE a missed expert is still in flight;
        # only the missed expert's own read has to wait for it to land.
        # Serialising the whole layer behind the fetch (the earlier model)
        # over-predicts by up to one layer-time per missing barrier -- ~1%
        # in decode, where compute is tens of us against ms-scale fetches,
        # but up to 10% in prefill, where a barrier's compute (~0.8 ms) is
        # comparable to a fetch. Validated over 65 gem5 replays: mean ratio
        # 0.9867 -> 0.9982, prefill 7/10 -> 10/10 inside +/-3%, with no
        # fitted parameter (MODEL.md 5b).
        t_layer = t_layer_fn(len(keys), _step)
        t_missed = nbytes / hbm_bps if nbytes else 0.0
        t_resident = max(0.0, layer_hbm / hbm_bps - t_missed)
        hbm_busy_s += layer_hbm / hbm_bps          # HBM streaming time
        cxl_busy_s += nbytes / link_bps            # expander serving time
        # t0 is per-barrier host overhead: serial, paid once, BEFORE either
        # path -- so it shifts the barrier start rather than sitting inside
        # one branch of the max (putting it only in the compute floor drops
        # it whenever the barrier is memory-bound).
        bstart = now + t0_s
        floor = (t_compute_fn(len(keys), _step) if t_compute_fn is not None
                 else t_layer - t0_s)
        begin = max(now, arrive)              # reported stall: unchanged
        stalled += begin - now
        layer_begin.append(begin)
        now = max(max(bstart + t_resident, arrive) + t_missed, bstart + floor)
        compute_s += t_layer
        policy.admit(keys, idx)

    n_layers = len(seq)
    return {
        "layers_executed": n_layers,
        "expert_requests": hits + misses,
        "hit_rate": hits / max(1, hits + misses),
        "bytes_fetched": fetched,
        "bytes_hbm_read": hbm_read,
        "bytes_hbm_write_kv": hbm_write,
        "total_s": now,
        "stall_s": stalled,
        "stall_fraction": stalled / max(1e-12, now),
        "compute_s": compute_s,
        # activity fractions: what share of wall time each tier spends
        # streaming. Feed to EnergyModel.energy(u_hbm=, u_ddr5=) for
        # state-resolved background power instead of the saturated endpoint.
        "u_hbm": min(1.0, hbm_busy_s / max(1e-12, now)),
        "u_ddr5": min(1.0, cxl_busy_s / max(1e-12, now)),
        "achieved_link_bps": fetched / max(1e-12, now),
    }


# --------------------------------------------------------------------------

def main():
    from gpu_model import GPUS, attn_quad_flops, peak_flops, t_layer as roofline_t

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--hbm-gib", type=float, default=24.0)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp8"],
                    help="fp8 halves every weight and doubles peak FLOPS")
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill"],
                    help="prefill = one barrier per layer over the whole "
                         "prompt; compute-bound and the harshest tiering case")
    ap.add_argument("--gpu", default="H100", choices=list(GPUS),
                    help="datasheet FLOPS + stack count anchor")
    ap.add_argument("--policy", default="lru",
                    choices=["none", "static", "lru", "lru_layer", "oracle"])
    ap.add_argument("--link-gbps", type=float, default=50.7,
                    help="measured effective CXL 2.0 x16 with DDR5-6400 "
                         "backend (sim/README.md); CXL3 x16 = 114.5 with "
                         "deep device FIFOs, PCIe4 x16 = 24.9. All three "
                         "re-measured at exact-nominal link stages")
    ap.add_argument("--hbm-gbps", type=float, default=736.6,
                    help="measured HBM3 per stack (sim/README.md)")
    ap.add_argument("--stacks", type=int, default=None,
                    help="HBM stacks; default = the chosen GPU's datasheet")
    ap.add_argument("--prefetch-depth", type=int, default=0,
                    help="layers of lookahead; 0 = demand fetch. Staged "
                         "experts occupy cache slots while in flight, so "
                         "depth REDUCES effective capacity by depth x top_k")
    ap.add_argument("--fifo-entries", type=int, default=None,
                    help="device FIFO depth per protocol bridge (8 bridges); "
                         "caps the effective link rate at the Little's-law "
                         "limit 8*entries*64B/RTT.  The CXL3 fetch rate of "
                         "114.5 GB/s needs ~128 entries; the silicon-validated "
                         "48 caps fetch traffic at ~101 GB/s (84%% of nominal). "
                         "Default: uncapped (deep-FIFO assumption).")
    ap.add_argument("--fifo-rtt-ns", type=float, default=242.0,
                    help="round trip used for the FIFO Little's-law cap. "
                         "SHALLOWER than the deep-FIFO loaded RTT (280 ns): "
                         "a 48-entry window queues less, so it turns over "
                         "faster. Measured 239-245 ns by bisection against "
                         "gem5 replays with real 48-entry bridges.")
    ap.add_argument("--t0-us", type=float, default=20.0,
                    help="per-barrier launch/dispatch overhead; bracketed "
                         "[5,40] us (see gpu_model.t_layer), sweep it")
    ap.add_argument("--xfer-lat-ns", type=float, default=280.0,
                    help="loaded CXL round-trip per barrier fetch: measured "
                         "added latency ~160 ns + expander DDR5 access; "
                         "280 ns is the loaded RTT sim/README.md uses for "
                         "Little's-law sizing")
    ap.add_argument("--t-layer-us", type=float, default=None,
                    help="override per-layer time; default is the "
                         "gpu_model.py roofline")
    ap.add_argument("--emit-schedule", default=None, metavar="PATH",
                    help="write the per-barrier byte schedule + analytic "
                         "prediction as JSON, for gem5 replay "
                         "(sim/configs/moe_replay.py)")
    ap.add_argument("--max-barriers", type=int, default=None,
                    help="truncate the sequence (replay runs are minutes "
                         "per barrier-hundred; analytic output then covers "
                         "the same truncated window)")
    ap.add_argument("--warmup-barriers", type=int, default=0,
                    help="feed this many barriers to the policy WITHOUT "
                         "timing them, then simulate the next max-barriers "
                         "with warm state. Cold-started online policies "
                         "otherwise miss everything in a short window. "
                         "Incompatible with --policy oracle (future "
                         "indices shift)")
    args = ap.parse_args()

    gpu = GPUS[args.gpu]
    stacks = args.stacks if args.stacks is not None else gpu.stacks
    dtype_bytes = 2 if args.dtype == "fp16" else 1
    phase = DECODE if args.phase == "decode" else 0

    seq, run = demand_sequence(args.tag, args.batch, phase)
    amap = AddressMap(args.tag, dtype_bytes, args.hbm_gib)
    g = amap.g

    hbm_bps = args.hbm_gbps * 1e9 * stacks
    link_bps = args.link_gbps * 1e9
    if args.fifo_entries:
        # Little's law: the link cannot carry more than the outstanding
        # window allows, whatever the wire rate.  8 protocol bridges each
        # hold fifo_entries requests of one 64 B line; the loaded round
        # trip is the same RTT the latency term uses.
        cap = 8 * args.fifo_entries * 64 / (args.fifo_rtt_ns * 1e-9)
        link_bps = min(link_bps, cap)

    base_bytes = amap.att_bytes + amap.shared_bytes
    base_params = base_bytes / dtype_bytes
    P = run["prompt_len"]
    tokens = args.batch if args.phase == "decode" else args.batch * P

    # KV-cache traffic and capacity.  KV lives in HBM: every decode barrier
    # reads the whole cache-so-far for its layer and appends one token; the
    # fully-grown pool is reserved out of HBM up front (KV competes with the
    # expert cache for the same capacity -- a real and often-ignored tension).
    kv_bpt = kv_bytes_per_tok_layer(amap.att, dtype_bytes)
    G = run.get("gen_len", 0)
    kv_pool = args.batch * (P + G) * kv_bpt * g["num_layers"]

    def kv_layer_fn(step: int):
        if args.phase == "decode":
            return (args.batch * (P + step) * kv_bpt,     # read all cached
                    args.batch * kv_bpt)                  # append one token
        return (0.0, args.batch * P * kv_bpt)             # prefill: write P

    t0 = args.t0_us * 1e-6
    quad = (attn_quad_flops(amap.att, args.batch, P)
            if args.phase == "prefill" else 0.0)
    if args.t_layer_us is not None:
        t_fixed = args.t_layer_us * 1e-6
        t_layer_fn = lambda n, s: t_fixed                    # noqa: E731
    else:
        def t_layer_fn(n_experts: int, step: int) -> float:
            kv_rd, kv_wr = kv_layer_fn(step)
            b = base_bytes + n_experts * amap.expert_bytes + kv_rd + kv_wr
            p = base_params + n_experts * g["expert_params"]
            return roofline_t(b, p, tokens, gpu, dtype_bytes, hbm_bps,
                              t0_s=t0, extra_flops=quad)

    def t_compute_fn(n_experts: int, step: int) -> float:
        """Pure compute time: FLOPs/peak. t0 is added by simulate() as a
        per-barrier serial term, so it must NOT be included here."""
        p = base_params + n_experts * g["expert_params"]
        return (2.0 * p * tokens + quad) / peak_flops(gpu, dtype_bytes)

    # 4a: prefetch staging cost -- in-flight experts need landing slots, so
    # lookahead shrinks the cache that made it unnecessary.  A real tension:
    # sweeping depth trades stall hiding against hit rate.
    kv_slots = int(kv_pool // amap.expert_stride) + (1 if kv_pool else 0)
    eff_slots = max(0, amap.cache_slots - kv_slots
                    - args.prefetch_depth * g["top_k"])
    W = args.warmup_barriers
    if W and args.policy == "oracle":
        raise SystemExit("--warmup-barriers is incompatible with oracle")
    if args.max_barriers:
        seq = seq[:W + args.max_barriers]
    pol = make_policy(args.policy, eff_slots, seq, len(g["moe_layers"]))
    if W:
        for idx, (_s, L, experts) in enumerate(seq[:W]):
            pol.admit([(L, e) for e in experts], idx)
        seq = seq[W:]
    barriers = [] if args.emit_schedule else None
    r = simulate(seq, pol, amap.expert_bytes, link_bps, t_layer_fn,
                 args.prefetch_depth, base_layer_bytes=base_bytes,
                 hbm_bps=hbm_bps, kv_layer_fn=kv_layer_fn, collect=barriers,
                 xfer_lat_s=args.xfer_lat_ns * 1e-9, t_compute_fn=t_compute_fn,
                 t0_s=t0)

    if args.emit_schedule:
        from gpu_model import peak_flops
        flops = peak_flops(gpu, dtype_bytes)
        for b in barriers:
            p = base_params + b["n_experts"] * g["expert_params"]
            b["t_compute_floor_s"] = t0 + (2.0 * p * tokens + quad) / flops
        json.dump({
            "tag": args.tag, "batch": args.batch, "phase": args.phase,
            "dtype": args.dtype, "policy": args.policy,
            "depth": args.prefetch_depth, "hbm_gib": args.hbm_gib,
            # replay builds the link at NOMINAL width; efficiency emerges.
            # analytic uses the measured EFFECTIVE rate.  Known pairs from
            # sim/system_params.py LINK_EFFICIENCY.
            "link_gbps_nominal": min((26.0, 63.0, 121.0),
                                     key=lambda n: abs(n * 0.8
                                                       - args.link_gbps)),
            "link_gbps_effective": args.link_gbps,
            "hbm_gbps_per_stack": args.hbm_gbps, "stacks": stacks,
            "gpu": args.gpu, "hbm_gen": gpu.hbm_gen,
            # u_hbm / u_ddr5 are duty cycles, carried so a replay's energy
            # can be scored against the SAME state-resolved background power
            # the sweep uses (energy_model.p_background).
            "analytic": {k: r[k] for k in
                         ("total_s", "stall_s", "compute_s", "hit_rate",
                          "bytes_fetched", "bytes_hbm_read",
                          "u_hbm", "u_ddr5")},
            "barriers": barriers,
        }, open(args.emit_schedule, "w"), indent=1)
        print(f"  schedule -> {args.emit_schedule} ({len(barriers)} barriers)")

    steps = len(seq) / max(1, len(g["moe_layers"]))
    t_first = t_layer_fn(g["top_k"], 0)
    print(f"=== {args.tag}  batch {args.batch}  {args.phase}  {args.dtype}  "
          f"HBM {args.hbm_gib}G  policy={args.policy}  "
          f"depth={args.prefetch_depth} ===")
    total_experts = g['num_experts'] * len(g['moe_layers'])
    slot_note = f" (kv pool -{kv_slots}"
    if args.prefetch_depth:
        slot_note += f", staging -{args.prefetch_depth * g['top_k']}"
    slot_note += ")"
    print(f"  expert {amap.expert_bytes/2**20:.1f} MiB   "
          f"cache {eff_slots}/{total_experts} experts "
          f"({100*min(1, eff_slots/total_experts):.0f}%){slot_note}   "
          f"KV {kv_pool/2**30:.2f} GiB ({kv_bpt} B/tok/layer)")
    print(f"  T_layer(k) {t_first*1e6:.1f} us   link {args.link_gbps} GB/s   "
          f"HBM {args.hbm_gbps*stacks:.0f} GB/s   {gpu.name} {args.dtype}")
    print(f"  --")
    print(f"  hit rate           {r['hit_rate']*100:>8.2f}%")
    print(f"  bytes fetched      {r['bytes_fetched']/GIB:>8.2f} GiB")
    print(f"  compute time       {r['compute_s']*1e3:>8.2f} ms")
    print(f"  stall time         {r['stall_s']*1e3:>8.2f} ms  "
          f"({r['stall_fraction']*100:.1f}% of total)")
    print(f"  total time         {r['total_s']*1e3:>8.2f} ms")
    print(f"  per decode step    {r['total_s']/steps*1e3:>8.3f} ms  "
          f"(ideal {r['compute_s']/steps*1e3:.3f} ms)")
    print(f"  slowdown vs all-resident {r['total_s']/max(1e-12,r['compute_s']):>5.2f}x")

    # ---- memory-side energy + EDP (see energy_model.py for provenance) ----
    from energy_model import EnergyModel, E_LINK_CTRL_SWEEP
    n_hbm_ch = stacks * 16
    print(f"  -- energy (memory side; {gpu.hbm_gen}, {n_hbm_ch} HBM ch + "
          f"8 DDR5 subch; e_link swept {E_LINK_CTRL_SWEEP} pJ/bit)")
    for e_link in E_LINK_CTRL_SWEEP:
        m = EnergyModel(gpu.hbm_gen, n_hbm_ch, 8, e_link)
        e = m.energy(r["bytes_hbm_read"],
                     r["bytes_fetched"] + r["bytes_hbm_write_kv"],
                     r["bytes_fetched"], r["total_s"],
                     u_hbm=r["u_hbm"], u_ddr5=r["u_ddr5"])
        print(f"    e_link={e_link:>4.0f}: total {e['e_total_j']:>7.3f} J  "
              f"(hbm {e['e_hbm_read_j']:.3f} + fill {e['e_hbm_fill_j']:.3f} "
              f"+ cxl {e['e_cxl_read_j']:.3f} + bg {e['e_background_j']:.3f})"
              f"  avg {e['avg_power_w']:>6.1f} W  "
              f"EDP {e['edp_js']:.4f} J*s")


if __name__ == "__main__":
    main()
