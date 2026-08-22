"""MoE decode replayed ON gem5: the host issues matmul memory traffic.

The professor's framing: model rough GPU-type matmul on gem5 and use it as
the compute + memory requests coming out of the host.  For decode, a GEMV's
memory behaviour IS the weight stream -- every touched weight is read once
per step -- so the host is modelled as request generators that, per layer
barrier:

    * stream that layer's HBM-resident bytes (attention + shared + resident
      experts + KV cache) across the HBM channels        <- "the matmul"
    * stream that layer's missed experts over the CXL path (link + protocol
      bridges + DDR5-6400 expander)                      <- the fetches

and the next layer cannot begin until BOTH are done (plus an analytic
compute floor for compute-bound barriers, e.g. prefill).  Both memory
systems are the calibrated ones from smoke_hbm.py / cxl_tier.py, unchanged.

The schedule (which bytes each barrier moves) comes from
`analytical/trace_gen.py --emit-schedule` -- i.e. from the real routing logs
with the policy already applied -- so this run is a CROSS-CHECK of the
analytic recurrence: gem5 decides all timing, the analytic model predicts
it, and they must agree within tolerance.

Co-simulation mechanics (all verified in isolation before this was built):
  * gem5's Python can interleave m5.simulate() with generator restarts as
    long as each trace ends with createExit()  -- restarting a generator
    whose trace has NOT exited segfaults gem5 25.1.
  * Every quantum (default 4 us) each active generator gets a fresh linear
    segment from its stream cursor; actual progress is read back from the
    memory controllers via m5.stats.gem5stats.get_simstat (no stats.txt
    churn) and cursors advance by measured bytes, so backpressure is
    honoured exactly.
  * Multiple generators exit at the same tick; the drive loop drains all
    exit events until the quantum boundary is reached.

Scaling: the HBM side models ONE stack (16 channels, the configuration the
bandwidth calibration validated); per-barrier HBM bytes are divided by the
stack count, which is exact for bandwidth-bound streaming across
independently-scaling stacks (linearity was measured, sim/README.md).

Run from the gem5 root:
    build/NULL/gem5.opt ../sim/configs/moe_replay.py --schedule sched.json
"""

import argparse
import json
import math
import os

import m5
from m5.objects import (AddrRange, Bridge, DRAMsim3, NoncoherentXBar,
                        PyTrafficGen, Root, SrcClockDomain, System,
                        VoltageDomain)

ap = argparse.ArgumentParser()
ap.add_argument("--schedule", required=True)
ap.add_argument("--hbm-config", default="HBM3_16Gb_x64_1ch")
ap.add_argument("--cxl-config", default="DDR5_6400_4Gb_x8")
ap.add_argument("--hbm-channels", type=int, default=4,
                help="HBM channels INSTANTIATED. The per-channel byte share "
                     "is always that of the real 16-channel stack, so this "
                     "samples channels rather than shrinking the stack -- "
                     "exact under the measured linear channel scaling, and "
                     "4x fewer simulation events than a full stack")
ap.add_argument("--cxl-channels", type=int, default=8)
ap.add_argument("--cxl-gens", type=int, default=8)
ap.add_argument("--quantum-ns", type=int, default=4000)
ap.add_argument("--bus-ghz", type=float, default=4.0)
ap.add_argument("--bridge-lat-ns", type=float, default=50.0)
ap.add_argument("--host-proto-ns", type=float, default=12.0)
ap.add_argument("--dev-proto-ns", type=float, default=15.0)
ap.add_argument("--fifo", type=int, default=48)
ap.add_argument("--fill-writes", action="store_true",
                help="also issue the cache-fill WRITE of each missed "
                     "expert into HBM. The analytic model charges this "
                     "(bytes_fetched x e_hbm_wr, and nbytes/hbm_bps on "
                     "the arrival path); without it a replay measures a "
                     "read-only system and its energy is short by the "
                     "fill term by construction. Off by default so the "
                     "read-only timing campaign stays reproducible.")
ap.add_argument("--max-barriers", type=int, default=None)
ap.add_argument("--energy-dir", default=None,
                help="give each DRAMsim3 instance its own output directory "
                     "so a full replay can report COMPOSED energy (HBM + "
                     "expander together). Without it every controller "
                     "overwrites one shared dramsim3.json.")
args = ap.parse_args()

sched = json.load(open(args.schedule))
barriers = sched["barriers"][:args.max_barriers or None]
STACKS = sched["stacks"]
LINK_GBPS = sched["link_gbps_nominal"]
DEPTH = sched.get("depth", 0)

DRAMSIM = "ext/dramsim3/DRAMsim3/"
PS = 1000                       # ticks (ps) per ns
QUANTUM = args.quantum_ns * PS

# --------------------------------------------------------------------------
# system: two port islands -- HBM pairs, and the CXL tier -- one host
# --------------------------------------------------------------------------
system = System()
system.clk_domain = SrcClockDomain(clock="2GHz", voltage_domain=VoltageDomain())
system.mem_mode = "timing"

HBM_SLICE = 1 << 30                                   # 1 GiB per channel
CXL_BASE = 1 << 40                                    # far away
CXL_SIZE = 16 << 30

system.mem_ranges = [AddrRange(0, size=args.hbm_channels * HBM_SLICE),
                     AddrRange(CXL_BASE, size=CXL_SIZE)]


# @tandonmitul27 -- per-instance energy output.
# DRAMsim3 writes its per-channel energy totals to "dramsim3.json" inside
# filePath when the wrapper is destroyed.  Every controller defaulting to
# the same directory means they overwrite each other, which is why energy
# could previously only be measured on single-instance runs.  With
# --energy-dir each controller gets its own subdirectory, so a FULL replay
# (HBM stack + CXL tier together) can report composed energy -- the only
# way to validate the energy model end to end rather than per device.
_ENERGY_DIRS = []


def dramsim(cfg, rng, tag=None):
    c = DRAMsim3()
    c.configFile = DRAMSIM + "configs/" + cfg + ".ini"
    if args.energy_dir and tag:
        d = os.path.join(args.energy_dir, tag)
        os.makedirs(d, exist_ok=True)
        c.filePath = d + "/"
        _ENERGY_DIRS.append((tag, d, cfg))
    else:
        c.filePath = DRAMSIM
    c.range = rng
    return c


# HBM: one validated stack -- generator<->channel pairs, no shared fabric
# (an XBar in front of a stack breaks scaling; sim/README.md).
system.hbm_ctrls = [
    dramsim(args.hbm_config, AddrRange(i * HBM_SLICE, size=HBM_SLICE),
            tag=f"hbm{i}")
    for i in range(args.hbm_channels)
]
system.hbm_gens = [PyTrafficGen(progress_check="100s")
                   for _ in range(args.hbm_channels)]
for g, c in zip(system.hbm_gens, system.hbm_ctrls):
    g.port = c.port

# CXL tier: cxl_tier.py topology verbatim -- aggregation bus, single-port
# link stage (width x clock = nominal GB/s), per-channel protocol bridges,
# interleaved DDR5-6400 backend.
busclk = SrcClockDomain(clock=f"{args.bus_ghz}GHz",
                        voltage_domain=VoltageDomain())
system.busclk = busclk
system.cxl_gens = [PyTrafficGen(progress_check="100s")
                   for _ in range(args.cxl_gens)]
system.aggbus = NoncoherentXBar(width=256, frontend_latency=1,
                                forward_latency=1, response_latency=1,
                                clk_domain=busclk)
for g in system.cxl_gens:
    g.port = system.aggbus.cpu_side_ports
system.system_port = system.aggbus.cpu_side_ports

system.link = NoncoherentXBar(
    width=max(1, int(round(LINK_GBPS / args.bus_ghz))),
    frontend_latency=1, forward_latency=1, response_latency=1,
    clk_domain=busclk)
system.link.cpu_side_ports = system.aggbus.mem_side_ports

proto_ns = args.bridge_lat_ns + args.host_proto_ns + args.dev_proto_ns
intlv_bits = int(math.log2(args.cxl_channels))
cxl_rng = system.mem_ranges[1]


def cxl_chan_range(i):
    return AddrRange(cxl_rng.start, size=cxl_rng.size(),
                     intlvHighBit=8 + intlv_bits - 1, xorHighBit=0,
                     intlvBits=intlv_bits, intlvMatch=i)


system.cxl_ctrls = [dramsim(args.cxl_config, cxl_chan_range(i),
                            tag=f"cxl{i}")
                    for i in range(args.cxl_channels)]
system.bridges = [
    Bridge(delay=f"{proto_ns}ns", req_size=args.fifo, resp_size=args.fifo,
           ranges=[c.range])
    for c in system.cxl_ctrls
]
for br, c in zip(system.bridges, system.cxl_ctrls):
    br.cpu_side_port = system.link.mem_side_ports
    br.mem_side_port = c.port

root = Root(full_system=False, system=system)
m5.instantiate()

from m5.stats.gem5stats import get_simstat    # noqa: E402  (after m5 init)


def ctrl_bytes(ctrl):
    """Cumulative bytes served (read+write) by one controller, live."""
    j = get_simstat(ctrl).to_json()
    tot = 0.0
    for key in ("bytesRead", "bytesWritten"):
        node = j.get(key, {}).get("value", {})
        tot += sum(v.get("value", 0.0) or 0.0
                   for v in node.values() if isinstance(v, dict))
    return tot


def gen_packets(g):
    """Packets ISSUED by a generator (sent, possibly still in flight).
    Address cursors advance by issued bytes, not completed bytes --
    advancing by controller counters re-issues the in-flight window every
    quantum (duplicate reads) and, worse, keeps all generators at the SAME
    span offset, which under 256B interleave makes them hammer one backend
    channel in lockstep (the same artifact the HBM calibration found)."""
    v = get_simstat(g).to_json().get("numPackets", {})
    if isinstance(v, dict):
        v = v.get("value", 0.0)
    if isinstance(v, dict):
        v = v.get("value", 0.0)
    return float(v or 0.0)


# --------------------------------------------------------------------------
# the drive loop
# --------------------------------------------------------------------------
def start_gen(g, cur, end, quantum, limit, rd_pct=100):
    """One quantum of sequential accesses, capped at `limit` bytes so a stream
    never overshoots its barrier target (it idles out the quantum after the
    cap -- harmless, the trace exits before the quantum boundary either way).

    `rd_pct` < 100 mixes in the cache-fill writes (--fill-writes): the fill
    lands on the same channels and the same rows as the reads it follows, so
    one mixed generator models it more faithfully than a separate write
    stream aimed at a disjoint region would.

    The trace must END strictly BEFORE the boundary: an exit event landing
    exactly on the boundary tick races the next start() and trips
    `assert !event->scheduled()` (double-schedule) in gem5's event queue.
    """
    seg = g.createLinear(quantum - 1000, cur, end, 64, 20, 20, int(rd_pct),
                         max(64, int(limit)))
    g.start([seg, g.createExit(0)])


def drain_to(target_tick):
    """Advance simulation to target_tick, absorbing generator exits."""
    while m5.curTick() < target_tick:
        m5.simulate(target_tick - m5.curTick())


hbm_base = [ctrl_bytes(c) for c in system.hbm_ctrls]
cxl_base = [ctrl_bytes(c) for c in system.cxl_ctrls]

# CXL fetch stream: one sequential cursor over the CXL region; the fetch for
# barrier i may begin once barrier max(0, i - DEPTH) has begun (the analytic
# issue rule).  cum_cxl[i] = stream position at which barrier i's data is
# fully arrived.
cum_cxl = []
acc = 0.0
for b in barriers:
    acc += b["cxl_bytes"]
    cum_cxl.append(acc)

results = []
cxl_pos = 0.0                                  # completed bytes (ctrls)
cxl_issued = 0.0                               # issued bytes (gens)
# per-gen address cursors, staggered by one interleave stride so the fetch
# generators start on different backend channels and stay de-synchronised
CXL_SPAN = (CXL_SIZE - (64 << 20)) // args.cxl_gens
cxl_cur = [j * 256 for j in range(args.cxl_gens)]
cxl_pkts0 = [gen_packets(g) for g in system.cxl_gens]
hbm_pos = [0.0] * args.hbm_channels            # completed bytes (ctrls)
hbm_cur = [0] * args.hbm_channels              # issued-byte cursors (gens)
hbm_pkts0 = [gen_packets(g) for g in system.hbm_gens]
n_hbm = args.hbm_channels

bi = 0                   # barrier being executed (compute side)
barrier_begin_tick = m5.curTick()
hbm_share = None
hbm_rd_pct = 100

while bi < len(barriers):
    b = barriers[bi]
    if hbm_share is None:
        # per-channel share of the REAL stack (16 ch x STACKS); only
        # n_hbm sampled channels are instantiated (see --hbm-channels)
        # the fill WRITE of every missed expert is HBM traffic too; the
        # analytic model always charged it, the replay only does so when
        # asked (--fill-writes).
        fill = b["cxl_bytes"] if args.fill_writes else 0.0
        hbm_share = (b["hbm_bytes"] + fill) / STACKS / 16
        hbm_rd_pct = round(100.0 * b["hbm_bytes"]
                           / max(1.0, b["hbm_bytes"] + fill))
        hbm_done_at = [p + hbm_share for p in hbm_pos]
        barrier_begin_tick = m5.curTick()

    # how far ahead the fetch stream may run: through barrier bi + DEPTH
    fetch_horizon = cum_cxl[min(bi + DEPTH, len(barriers) - 1)]

    # ---- start a quantum on every stream with eligible ISSUE work ----
    # issue caps come from issued bytes; completion below from controller
    # (completed) bytes -- the in-flight window is never re-issued.
    for i, g in enumerate(system.hbm_gens):
        to_issue = hbm_done_at[i] - (hbm_cur[i])
        if to_issue > 64:
            cur = int(hbm_cur[i]) % (HBM_SLICE - (64 << 20))
            start_gen(g, i * HBM_SLICE + (cur & ~63),
                      (i + 1) * HBM_SLICE, QUANTUM, to_issue, hbm_rd_pct)
    if cxl_issued < fetch_horizon - 64:
        per_gen = (fetch_horizon - cxl_issued) / args.cxl_gens
        for j, g in enumerate(system.cxl_gens):
            cur = int(cxl_cur[j]) % (CXL_SPAN - (1 << 20))
            start_gen(g, CXL_BASE + j * CXL_SPAN + (cur & ~63),
                      CXL_BASE + (j + 1) * CXL_SPAN, QUANTUM, per_gen)

    target = m5.curTick() + QUANTUM
    drain_to(target)

    # ---- read actual progress ----
    # completed (controllers) gates barriers; issued (generators) moves
    # the address cursors
    hbm_pos = [ctrl_bytes(c) - hbm_base[i]
               for i, c in enumerate(system.hbm_ctrls)]
    hbm_cur = [(gen_packets(g) - hbm_pkts0[i]) * 64
               for i, g in enumerate(system.hbm_gens)]
    cxl_pos = sum(ctrl_bytes(c) - cxl_base[i]
                  for i, c in enumerate(system.cxl_ctrls))
    cxl_new = [(gen_packets(g) - cxl_pkts0[j]) * 64
               for j, g in enumerate(system.cxl_gens)]
    cxl_cur = [j * 256 + b for j, b in enumerate(cxl_new)]
    cxl_issued = sum(cxl_new)

    # ---- barrier complete? both streams done + compute floor ----
    hbm_ok = all(hbm_pos[i] >= hbm_done_at[i] - 64 for i in range(n_hbm))
    cxl_ok = cxl_pos >= cum_cxl[bi] - 64
    floor_ticks = int(b.get("t_compute_floor_s", 0.0) * 1e12)
    floor_ok = m5.curTick() >= barrier_begin_tick + floor_ticks
    if hbm_ok and cxl_ok and floor_ok:
        results.append({"idx": bi, "step": b["step"], "layer": b["layer"],
                        "end_tick": m5.curTick()})
        bi += 1
        hbm_share = None

total_s = m5.curTick() / 1e12
analytic = sched["analytic"]["total_s"]
print(f"\nREPLAY barriers={len(results)} total={total_s*1e3:.3f} ms  "
      f"analytic={analytic*1e3:.3f} ms  "
      f"ratio={total_s/max(1e-12, analytic):.3f}")
out = {"replay_total_s": total_s, "analytic_total_s": analytic,
       "barriers": len(results), "quantum_ns": args.quantum_ns}

# @tandonmitul27 -- composed energy readout.
# DRAMsim3 flushes dramsim3.json when the wrapper is destroyed, so the files
# only exist after m5 tears down; record where to find them and let the
# caller aggregate. Units: the raw stats are V*mA*CYCLES, NOT pJ -- multiply
# by each config's tCK (sim/measure_energy.py, CALIBRATION.md "Units").
if args.energy_dir:
    out["energy_dirs"] = [{"tag": tg, "dir": d, "config": cfg}
                          for tg, d, cfg in _ENERGY_DIRS]
    out["hbm_scale"] = STACKS          # per-barrier bytes were divided by this
print(json.dumps(out))
