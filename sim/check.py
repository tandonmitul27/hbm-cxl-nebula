"""`make check` -- every validation this model's credibility rests on.

===========================================================================
 @tandonmitul27  --  AUTHORED FILE (new)
===========================================================================

WHY THIS FILE EXISTS
    "We validated it once, by hand, months ago" is not a claim anyone
    can act on.  This script re-runs every external comparison the model
    is built on, in one command, with a stated tolerance per check and a
    nonzero exit if any of them drifts.  It converts a past act of
    validation into a property of the current tree.

    It is also the first thing to run after `setup.sh`: if this passes,
    the build, the device configs, the harnesses and the parameters are
    all consistent with the published references.

WHAT IT COMPARES AGAINST -- all external, none self-referential
    bandwidth      HBM3 stack vs the NVIDIA H100 SXM5 datasheet
    latency        CXL added latency vs measured CXL ASIC and FPGA
                   silicon (CXL-DMSim Table II)
    fetch time     Mixtral expert over PCIe4 x16 vs FloE's published
                   ~15 ms
    energy         HBM pJ/bit vs O'Connor et al., MICRO'17; DDR5 pJ/bit
                   vs the DDR/GDDR device class
    self-consistency  row-miss penalty vs the tRP+tRCD configured in
                   the .ini, and address-map arithmetic vs published
                   checkpoint sizes

DESIGN RULE FOR ADDING CHECKS
    A check that cannot fail is not a check.  Tolerances are deliberately
    tight enough that a real regression trips them.

    python sim/check.py            # full  (~5 min; includes 16-inst runs)
    python sim/check.py --fast     # skips the two full-stack runs
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEM5 = ROOT / "gem5"          # created by setup.sh at the repo root
DRAMSIM = GEM5 / "ext" / "dramsim3" / "DRAMsim3"
GEM5_BIN = GEM5 / "build" / "NULL" / "gem5.opt"

ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = (f"{Path.home()}/miniconda3/lib:{DRAMSIM}:"
                          + ENV.get("LD_LIBRARY_PATH", ""))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, measured: float, lo: float, hi: float, unit: str,
          ref: str):
    ok = lo <= measured <= hi
    RESULTS.append((name, ok, ""))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:<44} {measured:>9.2f} {unit:<6} "
          f"(accept {lo:g}..{hi:g}; {ref})")
    return ok


def gem5(config_script: str, outdir: str, *args: str) -> Path:
    out = Path(f"/tmp/check-{outdir}")
    subprocess.run([str(GEM5_BIN), f"--outdir={out}",
                    str(HERE / "configs" / config_script), *args],
                   cwd=GEM5, env=ENV, capture_output=True, check=False)
    return out


def stat(outdir: Path, name: str, total=False) -> float:
    txt = (outdir / "stats.txt").read_text()
    vals = [float(l.split()[1]) for l in txt.splitlines()
            if l.startswith(name) or (total and re.match(
                name.replace("N", r"s?\d*") + r"\b", l.split()[0] if l.split() else ""))]
    return sum(vals) if total else (vals[0] if vals else float("nan"))


def stat_sum(outdir: Path, needle: str) -> float:
    txt = (outdir / "stats.txt").read_text()
    return sum(float(l.split()[1]) for l in txt.splitlines() if needle in l)


def ini_ns(config: str, *keys: str) -> float:
    """Sum of timing values (in ns) from a DRAMsim3 ini."""
    text = (DRAMSIM / "configs" / f"{config}.ini").read_text()
    tck = float(re.search(r"^tCK = ([\d.]+)", text, re.M).group(1))
    tot = 0.0
    for k in keys:
        cyc = float(re.search(rf"^{k} = (\d+)", text, re.M).group(1))
        tot += cyc * tck
    return tot


# ==========================================================================

def check_address_maps():
    """The address map must account for the whole model and no more.

    Independent reference: the published fp16 checkpoint size of each
    model.  If the map's region arithmetic drifts -- a miscounted
    attention projection, an alignment applied to a size instead of a
    base -- the total stops matching and this trips.  Tolerance is 1%,
    which is tight: the map currently agrees to within 0.3%.
    """
    print("\n== address map vs published checkpoint sizes ==")
    sys.path.insert(0, str(ROOT / "mapping"))
    from address_map import AddressMap, GIB
    published = {"OLMoE-1B-7B": 12.9, "DeepSeek-V2-Lite": 29.3,
                 "Phi-3.5-MoE": 78.0, "Mixtral-8x7B": 87.0}
    for tag, gb in published.items():
        m = AddressMap(tag, 2, 24.0)
        tot = m.summary()["total_model_bytes"] / GIB
        check(f"model bytes {tag}", tot, gb * 0.99, gb * 1.01, "GiB",
              "published fp16 checkpoint")


def check_row_miss():
    print("\n== HBM row-miss penalty vs configured tRP+tRCD ==")
    for cfg in ("HBM3_16Gb_x64_1ch", "HBM3e_24Gb_x64_1ch"):
        expect = ini_ns(cfg, "tRP", "tRCDRD")
        lat = {}
        for mode, extra in (("hit", []), ("miss", ["--random"])):
            o = gem5("smoke_hbm.py", f"rm-{cfg}-{mode}", "--config", cfg,
                     "--mem-size", "1GB", "--window-ns", "20000", "--direct",
                     "--max-outstanding", "1", *extra)
            lat[mode] = (stat(o, "simSeconds")
                         / stat_sum(o, "numReads::total") * 1e9)
        pen = lat["miss"] - lat["hit"]
        # open-row hits under random access dilute the average downward
        check(f"{cfg} row-miss penalty", pen, 0.80 * expect, 1.05 * expect,
              "ns", f"tRP+tRCD = {expect:.1f} ns")


def check_bandwidth(fast: bool):
    print("\n== bandwidth calibration ==")
    o = gem5("smoke_hbm.py", "bw1", "--config", "HBM3_16Gb_x64_1ch",
             "--mem-size", "1GB", "--window-ns", "20000", "--direct")
    bw = stat_sum(o, "bwRead::total") / 1e9
    check("HBM3 single channel sequential", bw, 40.0, 51.2, "GB/s",
          "88% of 51.2 peak measured")
    if fast:
        print("  [skip] full-stack runs (--fast)")
        return
    for cfg, n, lo, hi, ref in (
            ("HBM3_16Gb_x64_1ch", 16, 640, 780, "H100 datasheet 670 GB/s/stack"),
            ("HBM3e_24Gb_x64_1ch", 16, 930, 1130, "84% of 1229 peak measured")):
        o = gem5("smoke_hbm.py", f"stack-{cfg}", "--config", cfg,
                 "--mem-size", "2GB", "--window-ns", "20000",
                 "--pairs", str(n), "--sys-ghz", "8.0")
        bw = stat_sum(o, "bwRead::total") / 1e9
        check(f"{cfg} full stack ({n} ch)", bw, lo, hi, "GB/s", ref)


def check_cxl():
    print("\n== CXL path vs CXL-DMSim silicon measurements ==")
    base = ["--backend-channels", "8", "--num-gen", "1", "--bus-ghz", "4.0",
            "--max-outstanding", "1"]
    lat = {}
    for name, extra in (("local", ["--no-link"]),
                        ("asic", ["--link-gbps", "63"]),
                        ("fpga", ["--link-gbps", "63", "--dev-proto-ns", "60"])):
        o = gem5("cxl_tier.py", f"cxl-{name}", *base, *extra)
        lat[name] = (stat(o, "simSeconds")
                     / stat_sum(o, "numReads::total") * 1e9)
    check("CXL-ASIC added latency", lat["asic"] - lat["local"],
          154 * 0.88, 154 * 1.12, "ns", "silicon: 284-130 = 154 ns")
    check("CXL-FPGA added latency", lat["fpga"] - lat["local"],
          245 * 0.88, 245 * 1.12, "ns", "silicon: 375-130 = 245 ns")

    o = gem5("cxl_tier.py", "cxl-floe", "--backend-channels", "16",
             "--num-gen", "16", "--bus-ghz", "4.0", "--link-gbps", "26")
    bw = stat_sum(o, "bwRead::total") / 1e9
    ms = 352.3e6 / (bw * 1e9) * 1e3
    check("FloE Mixtral expert over PCIe4 x16", ms, 12.0, 18.0, "ms",
          "FloE reports ~15 ms")

    # CXL 3.0 x16 operating point.  Needs device buffering deeper than the
    # 2.0-ASIC 48 entries: 121 GB/s x ~280 ns RTT / 64 B ~ 530 outstanding,
    # so 48-entry FIFOs cap at ~86 GB/s (71%) -- a real finding, documented
    # in docs/CALIBRATION.md.  This point certifies the deep-buffer configuration.
    o = gem5("cxl_tier.py", "cxl3-x16", "--backend-channels", "8",
             "--num-gen", "16", "--bus-ghz", "16.0", "--link-gbps", "121",
             "--fifo", "128")
    bw = stat_sum(o, "bwRead::total") / 1e9
    check("CXL 3.0 x16 effective bandwidth", bw, 95.0, 121.0, "GB/s",
          "105.6 measured, 87% of 121 spec")


def check_fill_mix():
    print("\n== HBM read+write mix (cache-fill accounting) ==")
    # The timeline charges fills at bytes/BW_hbm on top of reads.  Certify:
    # a 90/10 read/write mix keeps ~95% of read-only throughput, so the
    # linear accounting is right to within the check tolerance.
    o = gem5("smoke_hbm.py", "fillmix", "--config", "HBM3_16Gb_x64_1ch",
             "--mem-size", "1GB", "--window-ns", "20000", "--direct",
             "--read-pct", "90")
    bw = (stat_sum(o, "bwRead::total") + stat_sum(o, "bwWrite::total")) / 1e9
    check("HBM3 90/10 mix total bandwidth", bw, 38.0, 46.0, "GB/s",
          "read-only 44.2; mix costs ~5% turnaround")


def check_energy():
    print("\n== memory energy vs published pJ/bit ==")
    # HBM bounds are tight because those currents are CALIBRATED to the
    # anchor; the check guards against config/wrapper regressions.  DDR5
    # currents are Micron-datasheet via gem5 (uncalibrated), checked against
    # the DDR/GDDR device-class range (O'Connor Fig.1a: GDDR5 14 pJ/bit).
    cases = [
        ("HBM3_16Gb_x64_1ch", 3.7, 4.3, "calibrated to O'Connor 3.97 pJ/bit"),
        ("HBM3e_24Gb_x64_1ch", 3.7, 4.3, "calibrated to O'Connor 3.97 pJ/bit"),
        ("DDR5_6400_4Gb_x8", 9.0, 16.0,
         "Micron IDD via gem5; GDDR5-class ~14 (O'Connor Fig.1a)"),
    ]
    stats_file = DRAMSIM / "dramsim3.json"
    for cfg, lo, hi, ref in cases:
        stats_file.unlink(missing_ok=True)
        o = gem5("smoke_hbm.py", f"energy-{cfg}", "--config", cfg,
                 "--mem-size", "1GB", "--window-ns", "20000", "--direct")
        if not stats_file.exists():
            RESULTS.append((f"{cfg} energy stats emitted", False, ""))
            print(f"  [FAIL] {cfg}: DRAMsim3 wrote no stats file"); continue
        raw = json.loads(stats_file.read_text())
        chans = list(raw.values()) if all(k.isdigit() for k in raw) else [raw]
        # DRAMsim3 energy stats are V*mA*CYCLES, not pJ: multiply by tCK
        # (ns).  See measure_energy.py for the derivation.
        tck = float(re.search(r"^tCK = ([\d.]+)",
                              (DRAMSIM / "configs" / f"{cfg}.ini")
                              .read_text(), re.M).group(1))
        tot = sum(float(c.get("total_energy", 0)) for c in chans) * tck
        bits = stat(o, "system.mem_ctrl.bytesRead::total") * 8
        check(f"{cfg} energy per bit", tot / bits, lo, hi, "pJ/bit", ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if not GEM5_BIN.exists():
        sys.exit("gem5 not built")

    check_address_maps()
    check_row_miss()
    check_bandwidth(a.fast)
    check_cxl()
    check_fill_mix()
    check_energy()

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'=' * 60}\n{n_ok}/{len(RESULTS)} checks passed")
    if n_ok != len(RESULTS):
        for name, ok, _ in RESULTS:
            if not ok:
                print(f"  FAILED: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
