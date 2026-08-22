"""CXL Type-3 memory expander tier, trace-driven.

System: the **GPU is the CXL host** (project clarification).  It has HBM on
package and a CXL link to a memory expander -- ONE hop, not the two-hop
CXL->CPU->PCIe->GPU path of shipping hardware.  This file models the CXL tier
only; the HBM tier is measured by smoke_hbm.py --pairs 16 (708 GB/s/stack).

    GPU ─┬─ HBM            local, see smoke_hbm.py
         └─ CXL link ── expander DRAM     this file

Topology (no CPU):

    tgen -> link  -> cxl_bridge -> dev_bus -> DRAMsim3 x N   (expander DRAM)
            ^^^^     ^^^^^^^^^^
            bandwidth cap       protocol latency

Modelled after CXL-DMSim / SimCXL, whose device model was validated against
real CXL FPGA and ASIC silicon at ~3.4% average error.  We run vanilla gem5
25.1 rather than the SimCXL fork (a gem5 23.1 fork, full-system oriented), so
the CXL path is reproduced from its published parameters rather than reusing
its SimObjects:

    host bridge   bridge_lat      50 ns
    host bridge   proto_proc_lat  12 ns
    device        proto_proc_lat  15 ns
    host link     req/rsp FIFO   128 entries
    device        req/rsp FIFO    48 entries

Their silicon measurements (LMbench random read, Fig. 9 / Table II):

    DDR-L  (local DDR5)  130 ns
    CXL-ASIC             284 ns   -> CXL path adds 154 ns
    CXL-FPGA             375 ns

154 ns is the quantity we can check: their 130 ns is full load-to-use
(including the LLC-miss path) while ours is memory-port latency, so only the
DELTA is comparable.  2 x (50 + 12 + 15) = 154 ns by construction.

The link is modelled as a bandwidth-capped crossbar: at a 1 GHz clock, a
datapath of W bytes/cycle is exactly W GB/s, so --link-gbps sets the width
directly.  Reference points:

    PCIe 4.0 x16   ~26 GB/s   (FloE's Mixtral validation target)
    PCIe 5.0 x16   ~63 GB/s   (CXL 2.0)
    PCIe 6.0 x16   ~121 GB/s  (CXL 3.0)

Run from the gem5 root:
    build/NULL/gem5.opt ../sim/configs/cxl_tier.py --link-gbps 63
    build/NULL/gem5.opt ../sim/configs/cxl_tier.py --local-ddr   # latency baseline
"""

import argparse

import m5
from m5.objects import (
    AddrRange, Bridge, DRAMsim3, NoncoherentXBar, PyTrafficGen, Root,
    SrcClockDomain, System, VoltageDomain,
)

ap = argparse.ArgumentParser()
ap.add_argument("--mem-config", default="DDR5_6400_4Gb_x8",
                help="expander backend DRAM; default is one DDR5-6400 32-bit "
                     "subchannel (authored from gem5's DDR5_6400_4x8, Micron "
                     "datasheet lineage -- see the ini header)")
ap.add_argument("--mem-size", default="4GB")
ap.add_argument("--backend-channels", type=int, default=8,
                help="DRAM subchannels behind the expander. 8 x DDR5-6400 "
                     "x32 = 204.8 GB/s = Marvell Structera X 2504 (4ch DDR5-"
                     "6400, '200 GB/s'); the LINK binds, as on real devices")
ap.add_argument("--intlv-bytes", type=int, default=256)
ap.add_argument("--link-gbps", type=float, default=63.0,
                help="link bandwidth; becomes the crossbar width at 1 GHz")
ap.add_argument("--bridge-lat-ns", type=float, default=50.0)
ap.add_argument("--host-proto-ns", type=float, default=12.0,
                help="CXL-DMSim Table III, ASIC config")
ap.add_argument("--dev-proto-ns", type=float, default=15.0)
ap.add_argument("--fifo", type=int, default=48,
                help="device req/rsp FIFO depth (CXL-DMSim ASIC = 48; the\n                      host-side link FIFOs are 128, so the device is the\n                      binding constraint and is what we model)")
ap.add_argument("--no-link", "--local-ddr", dest="local_ddr",
                action="store_true",
                help="control: the expander's OWN DRAM with the link uncapped "
                     "and protocol latency zeroed, isolating what the CXL path "
                     "costs. This is NOT 'local memory' -- in a GPU-host "
                     "system local memory is HBM, measured separately by "
                     "smoke_hbm.py. Topology is kept identical to the CXL case "
                     "so the harness's own limits cancel in the ratio.")
ap.add_argument("--window-ns", type=int, default=20_000)
ap.add_argument("--span-bytes", type=int, default=512 * 1024 * 1024)
ap.add_argument("--min-period", type=int, default=20)
ap.add_argument("--max-period", type=int, default=20)
ap.add_argument("--max-outstanding", type=int, default=0,
                help="1 = one request in flight, so latency = time / requests")
ap.add_argument("--bus-ghz", type=float, default=1.0,
                help="clock for the aggregation bus and link stage; link\n                      bandwidth = width x clock, so width scales with it")
ap.add_argument("--num-gen", type=int, default=8,
                help="concurrent requestors. A single generator through one\n                      port caps near 30 GB/s in gem5 regardless of link\n                      width; a real host has many outstanding requests.")
ap.add_argument("--random", action="store_true")
args = ap.parse_args()

DRAMSIM = "ext/dramsim3/DRAMsim3/"
PS_PER_NS = 1000

system = System()
system.clk_domain = SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange(args.mem_size)]
system.tgens = [PyTrafficGen(max_outstanding_reqs=args.max_outstanding)
                for _ in range(args.num_gen)]


def dramsim(rng):
    c = DRAMsim3()
    c.configFile = DRAMSIM + "configs/" + args.mem_config + ".ini"
    c.filePath = DRAMSIM
    c.range = rng
    return c


# Backend DRAM, interleaved so a sequential stream spreads across every
# channel.  Disjoint ranges would leave a linear sweep hammering one channel at
# a time, which looks exactly like a bandwidth ceiling.
import math

intlv_bits = int(math.log2(args.backend_channels))
low_bit = int(math.log2(args.intlv_bytes))
base = system.mem_ranges[0]


def chan_range(i):
    if args.backend_channels == 1:
        return AddrRange(base.start, size=base.size())
    return AddrRange(base.start, size=base.size(),
                     intlvHighBit=low_bit + intlv_bits - 1, xorHighBit=0,
                     intlvBits=intlv_bits, intlvMatch=i)


system.mem_ctrls = [dramsim(chan_range(i)) for i in range(args.backend_channels)]

# Baseline and CXL differ ONLY in link bandwidth and protocol latency; the
# topology, buffering and channel count are identical so that gem5's own
# arbitration and outstanding-request limits affect both equally and cancel in
# the ratio.
if args.local_ddr:
    link_gbps, proto_ns = 1e5, 0.0        # uncapped link, no protocol cost
else:
    link_gbps = args.link_gbps
    proto_ns = args.bridge_lat_ns + args.host_proto_ns + args.dev_proto_ns

busclk = SrcClockDomain(clock=f"{args.bus_ghz}GHz",
                        voltage_domain=VoltageDomain())
system.busclk = busclk

# gem5's XBar `width` is PER PORT, so a crossbar with many requestor ports
# models a switch, not a serial link.  Aggregate first, then cross a
# SINGLE-port stage whose one layer serializes everything: that is the link.
system.aggbus = NoncoherentXBar(width=256, frontend_latency=1,
                                forward_latency=1, response_latency=1,
                                clk_domain=busclk)
for g in system.tgens:
    g.port = system.aggbus.cpu_side_ports
system.system_port = system.aggbus.cpu_side_ports

system.link = NoncoherentXBar(
    width=max(1, int(round(link_gbps / args.bus_ghz))),
    frontend_latency=1, forward_latency=1, response_latency=1,
    clk_domain=busclk,
)
system.link.cpu_side_ports = system.aggbus.mem_side_ports

system.bridges = [
    Bridge(delay=f"{proto_ns}ns", req_size=args.fifo, resp_size=args.fifo,
           ranges=[c.range])
    for c in system.mem_ctrls
]
for br, c in zip(system.bridges, system.mem_ctrls):
    br.cpu_side_port = system.link.mem_side_ports
    br.mem_side_port = c.port

root = Root(full_system=False, system=system)
m5.instantiate()

# Each generator sweeps its own slice, staggered by one interleave stride so
# they do not march across the backend channels in lockstep.
span = args.span_bytes // args.num_gen
for gi, g in enumerate(system.tgens):
    mk = g.createRandom if args.random else g.createLinear
    start = gi * span + gi * args.intlv_bytes
    g.start([
        mk(args.window_ns * PS_PER_NS, start, start + span, 64,
           args.min_period, args.max_period, 100, 0),
        g.createExit(0),
    ])
exit_event = m5.simulate()
print(f"\nExiting @ tick {m5.curTick()} because {exit_event.getCause()}")
