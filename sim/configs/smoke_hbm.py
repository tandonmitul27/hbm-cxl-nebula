"""Near-tier (HBM) harness: PyTrafficGen -> DRAMsim3, no CPU.

===========================================================================
 @tandonmitul27  --  AUTHORED FILE (new; nothing upstream does this)
===========================================================================

WHY THIS FILE EXISTS
    Every HBM number the project quotes -- bandwidth, row-miss latency,
    pJ/bit -- is produced here.  gem5 ships example scripts for DRAM
    testing, but none of them (a) drive DRAMsim3 with the traffic shape
    an MoE expert fetch actually has, (b) let you build a full HBM stack
    out of independent per-channel controllers, or (c) expose the fabric,
    interleave and queue-depth knobs the measurement depends on.  Without
    this harness the device configs in configs/dramsim3/ would be
    unvalidated text.

WHAT IT ESTABLISHES
    1. gem5 NULL + DRAMsim3 runs and reports sane bandwidth
    2. createLinear reproduces an expert fetch (contiguous sequential
       sweep) -- the access pattern weight streaming actually generates
    3. per-channel bandwidth scales linearly (--pairs), which is what
       licenses quoting a stack figure as 16 x a channel figure
    4. row-miss penalty matches the configured tRP + tRCD, which is the
       check that the .ini timings are really in force

THE KNOB THAT MATTERS MOST (--direct / --pairs)
    A single crossbar in front of a whole stack silently costs ~16x of
    the bandwidth: gem5's XBar `width` is PER PORT, and one shared
    arbitration layer serialises every channel.  --direct removes the
    fabric; --pairs N builds N independent generator<->controller pairs.
    Anything that puts one XBar in front of a stack is measuring the
    crossbar, not the memory.  See docs/CALIBRATION.md.

USAGE -- run from the gem5 root (paths below are relative to it)
    build/NULL/gem5.opt ../sim/configs/smoke_hbm.py \
        --config HBM3_16Gb_x64_1ch --direct                  # one channel
    build/NULL/gem5.opt ../sim/configs/smoke_hbm.py \
        --config HBM3_16Gb_x64_1ch --pairs 16 --sys-ghz 8.0  # full stack
    build/NULL/gem5.opt ../sim/configs/smoke_hbm.py \
        --config HBM3_16Gb_x64_1ch --direct \
        --max-outstanding 1 --random                         # row-miss lat
    `make bw-stack` / `make row-miss` wrap the common ones.
===========================================================================
"""

import argparse
import math

import m5
from m5.objects import (
    AddrRange, DRAMsim3, NoncoherentXBar, PyTrafficGen, Root, SrcClockDomain,
    System, VoltageDomain,
)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="HBM3_16Gb_x64_1ch",
                help="DRAMsim3 config name under configs/ (see "
                     "configs/dramsim3/ in this repo)")
ap.add_argument("--mem-size", default="1GB")
ap.add_argument("--fetch-bytes", type=int, default=12 * 1024 * 1024,
                help="burst span; default = one OLMoE expert (12 MiB)")
ap.add_argument("--idle-ns", type=int, default=1000,
                help="idle gap between bursts, standing in for layer compute")
ap.add_argument("--bursts", type=int, default=1)
ap.add_argument("--window-ns", type=int, default=20_000,
                help="saturating read window per burst")
ap.add_argument("--min-period", type=int, default=20,
                help="ticks between requests (1 tick = 1 ps); small enough to "
                     "outrun the memory so the link saturates")
ap.add_argument("--max-period", type=int, default=20)
ap.add_argument("--max-outstanding", type=int, default=0,
                help="1 = one request in flight, so latency = time / requests")
ap.add_argument("--random", action="store_true",
                help="random access: exercises tRCD/tRP/tRAS/tFAW, which a "
                     "sequential stream leaves almost untested")
ap.add_argument("--direct", action="store_true",
                help="wire tgen straight to memory, bypassing any crossbar")
ap.add_argument("--xbar-width", type=int, default=64,
                help="crossbar datapath bytes/cycle (stock SystemXBar is 16, "
                     "which caps a 64 B line at 16 GB/s)")
ap.add_argument("--xbar-ghz", type=float, default=1.0)
ap.add_argument("--sys-ghz", type=float, default=1.0,
                help="tgen clock. It issues at most one 64 B request per\n                      cycle, so 1 GHz is itself a 64 GB/s ceiling.")
ap.add_argument("--num-inst", type=int, default=1,
                help="N interleaved DRAMsim3 instances. Each carries its own "
                     "outstanding-request budget, so this tests whether "
                     "per-channel bandwidth really scales linearly.")
ap.add_argument("--pairs", type=int, default=0,
                help="N independent generator<->memory pairs, NO shared\n                      fabric. Isolates memory-model scaling from crossbar\n                      effects: this must scale linearly by construction.")
ap.add_argument("--num-gen", type=int, default=1,
                help="concurrent traffic generators. One generator is packet-\n                      rate limited (~128 GB/s), well under an HBM3 stack; a\n                      real GPU has many concurrent requestors anyway.")
ap.add_argument("--block-bytes", type=int, default=64,
                help="request size. 64 = cacheline; bulk weight streaming is\n                      realistically larger, and a single generator is packet-\n                      rate limited, so this lifts the harness ceiling.")
ap.add_argument("--intlv-bytes", type=int, default=256,
                help="address interleaving granularity across instances")
ap.add_argument("--read-pct", type=int, default=100,
                help="read percentage; <100 mixes writes in, for validating "
                     "the fill write-bandwidth assumption (weights are RO, "
                     "but cache FILLS write into HBM)")
ap.add_argument("--replicate-example", action="store_true",
                help="run gem5's own dramsys.py traffic verbatim, as a control")
args = ap.parse_args()

DRAMSIM = "ext/dramsim3/DRAMsim3/"
CFG = DRAMSIM + "configs/" + args.config + ".ini"

system = System()
system.clk_domain = SrcClockDomain(clock=f"{args.sys_ghz}GHz",
                                   voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange(args.mem_size)]
system.tgens = [PyTrafficGen(max_outstanding_reqs=args.max_outstanding)
                for _ in range(args.num_gen)]
tg0 = system.tgens[0]


def dramsim(rng):
    c = DRAMsim3()
    c.configFile, c.filePath, c.range = CFG, DRAMSIM, rng
    return c


if args.pairs > 0:
    # N disjoint (generator, memory) pairs, each on its own address range and
    # directly connected.  There is no fabric to contend for, so any departure
    # from linear scaling here is the memory model itself.
    system.tgens = [PyTrafficGen(max_outstanding_reqs=args.max_outstanding)
                    for _ in range(args.pairs)]
    tg0 = system.tgens[0]
    per = system.mem_ranges[0].size() // args.pairs
    system.mem_ctrls = [
        dramsim(AddrRange(start=i * per, size=per)) for i in range(args.pairs)
    ]
    for g, c in zip(system.tgens, system.mem_ctrls):
        g.port = c.port
elif args.num_inst > 1:
    # Interleave N instances across the address space, as gem5's own MemConfig
    # builds a multi-channel memory.  gem5's DRAMsim3 wrapper caps TOTAL
    # outstanding requests at trans_queue_size (a scalar, not scaled by channel
    # count), so a single instance goes latency-bound well below the device's
    # capability.  N instances give N times the request budget -- which is what
    # makes the per-channel scaling assumption testable.
    assert args.num_inst & (args.num_inst - 1) == 0, "num-inst must be 2^k"
    intlv_bits = int(math.log2(args.num_inst))
    low_bit = int(math.log2(args.intlv_bytes))
    system.membus = NoncoherentXBar(
        width=args.xbar_width, frontend_latency=1, forward_latency=1,
        response_latency=1,
        clk_domain=SrcClockDomain(clock=f"{args.xbar_ghz}GHz",
                                  voltage_domain=VoltageDomain()),
    )
    for g in system.tgens:
        g.port = system.membus.cpu_side_ports
    system.system_port = system.membus.cpu_side_ports
    base = system.mem_ranges[0]
    system.mem_ctrls = [
        dramsim(AddrRange(base.start, size=base.size(),
                          intlvHighBit=low_bit + intlv_bits - 1, xorHighBit=0,
                          intlvBits=intlv_bits, intlvMatch=i))
        for i in range(args.num_inst)
    ]
    for c in system.mem_ctrls:
        c.port = system.membus.mem_side_ports
else:
    system.mem_ctrl = dramsim(system.mem_ranges[0])
    if args.direct:
        tg0.port = system.mem_ctrl.port              # isolates the memory
    else:
        system.membus = NoncoherentXBar(
            width=args.xbar_width, frontend_latency=1, forward_latency=1,
            response_latency=1)
        for g in system.tgens:
            g.port = system.membus.cpu_side_ports
        system.mem_ctrl.port = system.membus.mem_side_ports
        system.system_port = system.membus.cpu_side_ports

root = Root(full_system=False, system=system)
m5.instantiate()

PS_PER_NS = 1000


def workload(gen, gi):
    if args.replicate_example:
        # Verbatim traffic from gem5's configs/example/dramsys.py -- separates
        # "our parameters are wrong" from "our wiring is wrong".
        return [
            gen.createLinear(10000000, 0, 16777216, 64, 500, 1500, 65, 0),
            gen.createIdle(100000),
            gen.createExit(0),
        ]
    # Segments are TIME-boxed, not work-boxed: once data_limit is reached,
    # linear_gen returns MaxTick and the segment idles out the rest of its
    # duration, tripping the 1 ms progress watchdog.  So saturate for a fixed
    # window and measure achieved bandwidth instead -- which is what the
    # analytic model needs (fetch time = expert bytes / achieved bandwidth).
    mk = gen.createRandom if args.random else gen.createLinear
    # In pairs mode each generator must stay inside ITS OWN memory's
    # range, which is carved from mem_size, not fetch_bytes.
    span = (int(system.mem_ranges[0].size()) // args.pairs if args.pairs
            else args.fetch_bytes // args.num_gen)
    segs = []
    for i in range(args.bursts):
        # Stagger each generator by one interleave stride so generator i
        # begins on instance i.  Without this every generator marches
        # across the instances in lockstep and they all hammer the same
        # one at a time -- which looks exactly like poor memory scaling.
        base_off = gi * span
        start = (base_off if args.pairs else
                 base_off + gi * args.intlv_bytes)
        start += i * (span // max(1, args.bursts))
        segs.append(mk(args.window_ns * PS_PER_NS,
                       start, start + span,
                       args.block_bytes,
                       args.min_period, args.max_period,
                       args.read_pct,            # default 100: weights are RO
                       0))                       # duration bounds the burst
        segs.append(gen.createIdle(args.idle_ns * PS_PER_NS))
    segs.append(gen.createExit(0))
    return segs


for gi, g in enumerate(system.tgens):
    g.start(workload(g, gi))
exit_event = m5.simulate()
print(f"\nExiting @ tick {m5.curTick()} because {exit_event.getCause()}")
