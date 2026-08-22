#!/usr/bin/env python
"""Regression check for the dramsim3.cc flow-control patch.

The patch tightens gem5's admission test for DRAMsim3. It is argued to be a
no-op for read-only traffic (an outstanding read is held longer by gem5 than
by DRAMsim3, so the aggregate test already implies the per-queue one), but
every one of the 102 validated points was measured on the OLD binary. An
argument is not a measurement: re-run a spread of read-only points and
require the ratios to reproduce exactly.
"""
import csv, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

SV = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repo root, not one host
GEM5 = f"{ROOT}/sim/gem5"
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = (os.path.expanduser("~/miniconda3/lib")
                          + f":{GEM5}/ext/dramsim3/DRAMsim3:"
                          + ENV.get("LD_LIBRARY_PATH", ""))

# name -> (hbm_cfg, fifo, bus).  Chosen to span both HBM generations, all
# three links, both phases, and an all-hit control.
POINTS = {
    "PF2-phi-cxl3":     ("HBM3_16Gb_x64_1ch",   128, 15.125),
    "PF2-mixtral-50":   ("HBM3_16Gb_x64_1ch",    48,  4.0),
    "BB-phi-hr":        ("HBM3e_24Gb_x64_1ch",   48,  4.0),
    "PF2-olmoe-pcie4":  ("HBM3_16Gb_x64_1ch",    48,  6.5),
    "PF2-olmoe-h200":   ("HBM3e_24Gb_x64_1ch",   48,  4.0),
    "BB-deepseek":      ("HBM3e_24Gb_x64_1ch",   48,  4.0),
}


def prior():
    was = {}
    for f in ['results.csv', 'aux-results.csv', 'cxl3-results.csv',
              'gap-results.csv', 'fill-results.csv']:
        p = f'{SV}/{f}'
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if r['ratio'] != 'FAIL':
                    was[r['name']] = float(r['ratio'])
    return was


def run(item):
    name, (cfg, fifo, bus) = item
    out = f"{SV}/runs/RG-{name}"
    os.makedirs(out, exist_ok=True)
    cmd = [f"{GEM5}/build/NULL/gem5.opt", f"--outdir={out}",
           f"{ROOT}/sim/configs/moe_replay.py",
           "--schedule", f"{SV}/sched/{name}.json",
           "--hbm-config", cfg, "--fifo", str(fifo), "--bus-ghz", str(bus)]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=GEM5, env=ENV)
    m = re.search(r"ratio=([\d.]+)", r.stdout)
    return name, (float(m.group(1)) if m else None), (time.time() - t) / 60


if __name__ == "__main__":
    was = prior()
    with ThreadPoolExecutor(max_workers=6) as ex:
        got = list(ex.map(run, POINTS.items()))
    bad = 0
    print(f"\n{'point':20s} {'before':>8s} {'after':>8s} {'delta':>9s}")
    for name, now, mins in sorted(got):
        b = was.get(name)
        if now is None:
            print(f"{name:20s} {b:8.4f}    FAILED"); bad += 1; continue
        d = now - b
        flag = "" if abs(d) < 5e-4 else "   <-- CHANGED"
        bad += bool(flag)
        print(f"{name:20s} {b:8.4f} {now:8.4f} {d:+9.5f}{flag}"
              f"   ({mins:.1f} min)")
    print("\nREGRESSION " + ("FAILED" if bad else "CLEAN"))
    sys.exit(1 if bad else 0)
