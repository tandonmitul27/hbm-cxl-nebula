"""`make check` -- every validation the model's credibility rests on, one run.

Any change to configs, harnesses or parameters must keep this green.  Each
check prints measured vs expected with its tolerance; exit code is nonzero if
anything fails.  Tolerances are stated per check and err on the tight side --
a check that cannot fail is not a check.

    python sim/check.py            # full  (~5 min: includes 16-instance runs)
    python sim/check.py --fast     # skips the two full-stack bandwidth runs
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
# setup.sh builds gem5 at the REPO ROOT (see .gitignore /gem5/); a
# sibling-of-sim/ checkout is also honoured so a working tree that
# keeps it under sim/ still runs.
GEM5 = ROOT / "gem5" if (ROOT / "gem5").exists() else HERE / "gem5"
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
    print("\n== address map vs actual checkpoint sizes ==")
    sys.path.insert(0, str(ROOT / "mapping"))     # placement, not the model
    from address_map import AddressMap, GIB
    actual = {"OLMoE-1B-7B": 12.9, "DeepSeek-V2-Lite": 29.3,
              "Phi-3.5-MoE": 78.0, "Mixtral-8x7B": 87.0}
    for tag, gb in actual.items():
        m = AddressMap(tag, 2, 24.0)
        tot = m.summary()["total_model_bytes"] / GIB
        check(f"model bytes {tag}", tot, gb * 0.99, gb * 1.01, "GiB",
              "downloaded checkpoint")


def check_routing_integrity():
    print("\n== routing log integrity ==")
    r = subprocess.run([sys.executable, str(ROOT / "data" / "check_routing.py")],
                       capture_output=True, text=True)
    ok = "ALL CHECKS PASSED" in r.stdout
    RESULTS.append(("routing integrity (20 files)", ok, ""))
    print(f"  [{'PASS' if ok else 'FAIL'}] routing integrity (20 files, 17.1M records)")


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
    # Band tightened after the idle-gap dilution fix. NOTE the old band
    # (40.0-51.2) PASSED the diluted 44.24 GB/s figure -- a check calibrated
    # from the same buggy measurement confirms the bug. 46.04 = 90% of the
    # 51.2 GB/s per-channel peak.
    check("HBM3 single channel sequential", bw, 44.5, 51.2, "GB/s",
          "90% of 51.2 peak measured (46.04)")
    if fast:
        print("  [skip] full-stack runs (--fast)")
        return
    for cfg, n, lo, hi, ref in (
            ("HBM3_16Gb_x64_1ch", 16, 700, 790, "736.6 measured; H100 "
             "datasheet implies 670 GB/s/stack at shipping clocks"),
            ("HBM3e_24Gb_x64_1ch", 16, 1020, 1150, "1072.1 measured, "
             "87% of the 1229 spec peak")):
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

    # --bus-ghz 6.5 makes the link stage width 4 B x 6.5 GHz = EXACTLY the
    # 26.0 GB/s nominal, and 64 B / 4 B = 16 whole cycles.  At the old
    # --bus-ghz 4.0 the stage was width 6, so a 64 B packet needed
    # ceil(64/6) = 11 whole cycles -> a 23.3 GB/s CEILING, and the check
    # measured the crossbar rather than the device (the same failure the
    # HBM calibration found, one level up).
    o = gem5("cxl_tier.py", "cxl-floe", "--backend-channels", "16",
             "--num-gen", "16", "--bus-ghz", "6.5", "--link-gbps", "26")
    bw = stat_sum(o, "bwRead::total") / 1e9
    ms = 352.3e6 / (bw * 1e9) * 1e3
    check("FloE Mixtral expert over PCIe4 x16", ms, 12.0, 18.0, "ms",
          "FloE reports ~15 ms; ours ~13.9 at 25.3 GB/s steady state -- "
          "the gap is host software overhead FloE includes")

    # CXL 3.0 x16 operating point.  Needs device buffering deeper than the
    # 2.0-ASIC 48 entries: 121 GB/s x ~280 ns RTT / 64 B ~ 530 outstanding.
    # --bus-ghz 15.125 makes the stage width 8 B x 15.125 GHz = EXACTLY the
    # 121.0 GB/s nominal (64 B / 8 B = 8 whole cycles).  --bus-ghz 16.0
    # would give width round(121/16) = 8 at 16 GHz = 128 GB/s, a stage that
    # OVERSHOOTS the FLIT-corrected nominal and lets fetch traffic measure
    # faster than the link can physically run.
    o = gem5("cxl_tier.py", "cxl3-x16", "--backend-channels", "8",
             "--num-gen", "16", "--bus-ghz", "15.125", "--link-gbps", "121",
             "--fifo", "128")
    bw = stat_sum(o, "bwRead::total") / 1e9
    check("CXL 3.0 x16 effective bandwidth", bw, 95.0, 121.0, "GB/s",
          "106.6 measured at the exact-121 stage, 88% of spec (steady "
          "state; MoE fetch streams reach 117 -- system_params "
          "cxl3_x16_121_fetch)")


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
        # (ns).  See measure_energy.py for the derivation of this unit trap.
        tck = float(re.search(r"^tCK = ([\d.]+)",
                              (DRAMSIM / "configs" / f"{cfg}.ini")
                              .read_text(), re.M).group(1))
        tot = sum(float(c.get("total_energy", 0)) for c in chans) * tck
        bits = stat(o, "system.mem_ctrl.bytesRead::total") * 8
        check(f"{cfg} energy per bit", tot / bits, lo, hi, "pJ/bit", ref)


def check_recurrence_invariants():
    """Structural invariants of the barrier recurrence -- pure arithmetic,
    no gem5 needed, runs in ~2 s.

    These exist because the 75-point replay campaign is run t0-FREE (t0 is
    exact serial arithmetic on both sides, so including it would measure
    bookkeeping rather than the memory recurrence). That convention makes
    the campaign blind to any t0 handling bug -- and one shipped: folding
    t0 into the compute floor silently dropped it on memory-bound barriers,
    producing total_s BELOW the compute floor. No replay point could have
    caught it. These invariants can.
    """
    import subprocess, json, tempfile, os
    print("\n== recurrence invariants (t0 accounting, monotonicity) ==")
    PY = sys.executable
    ROOT = HERE.parent
    def run(tag, extra=()):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = f.name
        cmd = [PY, str(ROOT / "analytical" / "trace_gen.py"), "--tag", tag,
               "--batch", "16", "--hbm-gib", "80", "--policy", "static",
               "--max-barriers", "32", "--emit-schedule", out, *extra]
        subprocess.run(cmd, capture_output=True, cwd=str(ROOT))
        d = json.load(open(out))["analytic"]; os.unlink(out)
        return d
    for tag in ("OLMoE-1B-7B", "Phi-3.5-MoE"):
        d = run(tag)
        # 1. a run can never finish before its own compute floor
        # lower bound allows float noise from summing hundreds of barriers
        # (observed |delta| ~1e-17 s) while still catching a real violation,
        # which would be percent-scale -- the t0 bug this guards against
        # produced total_s ~8% BELOW the floor.
        check(f"{tag}: total >= compute", d["total_s"] / max(1e-12, d["compute_s"]),
              1.0 - 1e-9, 1e9, "x",
              "total_s must never fall below the compute floor")
        # 2. t0 must lengthen the run by exactly n_barriers * delta-t0
        a0 = run(tag, ("--t0-us", "0"))["total_s"]
        a20 = run(tag, ("--t0-us", "20"))["total_s"]
        got = (a20 - a0) * 1e6 / 32.0
        check(f"{tag}: t0 is serial per barrier", got, 19.0, 21.0, "us",
              "20 us of t0 must add exactly 20 us per barrier, whatever binds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if not GEM5_BIN.exists():
        sys.exit("gem5 not built")

    check_address_maps()
    check_routing_integrity()
    check_row_miss()
    check_bandwidth(a.fast)
    check_cxl()
    check_fill_mix()
    check_energy()
    check_recurrence_invariants()

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'=' * 60}\n{n_ok}/{len(RESULTS)} checks passed")
    if n_ok != len(RESULTS):
        for name, ok, _ in RESULTS:
            if not ok:
                print(f"  FAILED: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
