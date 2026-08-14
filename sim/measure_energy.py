"""Memory-side energy in pJ/bit, measured rather than assumed.

===========================================================================
 @tandonmitul27  --  AUTHORED FILE (new)
===========================================================================

WHY THIS FILE EXISTS
    gem5 does not surface DRAMsim3's energy statistics at all.  DRAMsim3
    computes them and writes them to its own dramsim3.json when the
    wrapper is destroyed -- into the DRAMSIM directory, NOT the gem5
    output directory, which is why they are easy to miss entirely.
    This script runs a configuration, finds that file, applies the unit
    correction below, and reports pJ/bit: the unit published DRAM energy
    figures are quoted in, and therefore the only unit in which our
    numbers can be checked against the literature.

UNITS -- the most important thing in this file
    DRAMsim3's energy stats are V * mA * CYCLES, not picojoules.  The
    energy increments carry timing in clock cycles and nothing in the
    energy path ever multiplies by tCK (only average_power is
    unit-correct, because the cycles cancel there).  True pJ =
    reported * tCK_ns.
    At the stock DDR4 tCK = 0.63 ns the discrepancy is small enough to
    overlook; at our HBM3 tCK = 0.3125 ns it is 3.2x.  This script is
    therefore the only supported way to read energy out of this repo.

USAGE
    python sim/measure_energy.py --config HBM3_16Gb_x64_1ch
    python sim/measure_energy.py --config HBM3_16Gb_x64_1ch \
                                 --config DDR5_6400_4Gb_x8
    make energy
===========================================================================

Original notes follow.
---------------------------------------------------------------------------

DRAMsim3 already computes energy the standard way -- IDD currents x VDD x time
in each state (activate, column access, refresh, standby).  gem5 does not
surface those stats, but DRAMsim3 writes them itself to
`ext/dramsim3/DRAMsim3/dramsim3.json` when the wrapper is destroyed.  This
script runs a configuration, reads that file, and converts to **pJ/bit** --
the unit published HBM/DDR energy figures are quoted in, so the result can be
checked against real data.

Energy of a CXL access has three terms and DRAMsim3 only models the first:

    E_cxl = E_dram  +  E_serdes  +  E_controller
            ^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^
            measured   analytic (no DRAM simulator models a serial link)

so the link and controller terms are parameters, declared in system_params.py
with their sources, and swept.

    python measure_energy.py --config HBM3_16Gb_x64_1ch
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEM5 = ROOT / "gem5"          # created by setup.sh at the repo root
DRAMSIM = GEM5 / "ext" / "dramsim3" / "DRAMsim3"
STATS = DRAMSIM / "dramsim3.json"

ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = (f"{Path.home()}/miniconda3/lib:{DRAMSIM}:"
                          + ENV.get("LD_LIBRARY_PATH", ""))

# UNITS: DRAMsim3's energy stats are V * mA * CYCLES, not pJ.  Its energy
# increments (configuration.cc InitPowerParams) carry timing in clock cycles
# and no tCK multiplication happens anywhere in the energy path -- only
# average_power is unit-correct (mW), because the cycles cancel there.
# True energy in pJ = reported * tCK_ns.  At the stock DDR4 tCK=0.63 or
# HBM2's 1.0 this is a benign scale; at our HBM3 tCK=0.3125 it is 3.2x.
# Everything below converts to pJ via the config's own tCK.
ENERGY_KEYS = ["act_energy", "read_energy", "write_energy", "ref_energy",
               "refb_energy"]
STANDBY_KEYS = ["act_stb_energy", "pre_stb_energy", "sref_energy"]


def config_tck(config: str) -> float:
    import re
    text = (DRAMSIM / "configs" / f"{config}.ini").read_text()
    return float(re.search(r"^tCK = ([\d.]+)", text, re.M).group(1))


def _flatten(v):
    """Standby energies are per-rank dicts; sum them."""
    if isinstance(v, dict):
        return sum(float(x) for x in v.values())
    return float(v)


def run(config: str, window_ns: int, extra: list[str]) -> dict:
    STATS.unlink(missing_ok=True)
    cmd = [str(GEM5 / "build" / "NULL" / "gem5.opt"), "--outdir=/tmp/energy",
           str(HERE / "configs" / "smoke_hbm.py"), "--config", config,
           "--mem-size", "1GB", "--window-ns", str(window_ns), "--direct",
           *extra]
    subprocess.run(cmd, cwd=GEM5, env=ENV, capture_output=True)
    if not STATS.exists():
        raise SystemExit("DRAMsim3 wrote no stats; did the run fail?")

    raw = json.loads(STATS.read_text())
    # One object per channel, keyed by channel id.
    chans = list(raw.values()) if all(k.isdigit() for k in raw) else [raw]

    tck = config_tck(config)                    # V*mA*cycles -> pJ
    tot = {k: sum(_flatten(c.get(k, 0)) for c in chans) * tck
           for k in ENERGY_KEYS + STANDBY_KEYS}
    tot["total_energy"] = sum(_flatten(c.get("total_energy", 0))
                              for c in chans) * tck
    tot["num_reads"] = sum(_flatten(c.get("num_read_cmds", 0)) for c in chans)
    tot["average_power_mW"] = sum(_flatten(c.get("average_power", 0))
                                  for c in chans)

    gstats = Path("/tmp/energy/stats.txt")
    txt = gstats.read_text() if gstats.exists() else ""
    def stat(name, default=0.0):
        for line in txt.splitlines():
            if line.startswith(name):
                return float(line.split()[1])
        return default
    tot["bytes_read"] = stat("system.mem_ctrl.bytesRead::total")
    tot["sim_seconds"] = stat("simSeconds")
    return tot


def report(name: str, t: dict):
    bits = t["bytes_read"] * 8
    if bits == 0:
        print(f"  {name}: no traffic recorded"); return
    dyn = sum(t[k] for k in ENERGY_KEYS)
    stb = sum(t[k] for k in STANDBY_KEYS)
    total = t["total_energy"] or (dyn + stb)
    print(f"  {name}")
    print(f"    bytes read          {t['bytes_read']/2**20:>10.2f} MiB")
    print(f"    activate            {t['act_energy']/1e6:>10.3f} uJ")
    print(f"    column read         {t['read_energy']/1e6:>10.3f} uJ")
    print(f"    refresh             {t['ref_energy']/1e6:>10.3f} uJ")
    print(f"    standby             {stb/1e6:>10.3f} uJ")
    print(f"    TOTAL               {total/1e6:>10.3f} uJ")
    print(f"    -> dynamic          {dyn/bits:>10.3f} pJ/bit")
    print(f"    -> total            {total/bits:>10.3f} pJ/bit"
          f"   (O'Connor MICRO'17: HBM 3.97 pJ/bit)")
    print(f"    avg power           {t['average_power_mW']:>10.1f} mW/channel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    help="DRAMsim3 config name; repeat to compare")
    ap.add_argument("--window-ns", type=int, default=20_000)
    ap.add_argument("--random", action="store_true",
                    help="row-miss dominated: activate energy grows")
    a = ap.parse_args()
    extra = ["--random"] if a.random else []
    print(f"=== memory-side energy ({'random' if a.random else 'sequential'}) ===")
    for c in a.config:
        report(c, run(c, a.window_ns, extra))


if __name__ == "__main__":
    main()
