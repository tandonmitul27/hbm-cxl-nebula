#!/usr/bin/env python
"""Static-placement validation grid: analytic schedule -> gem5 replay -> ratio.

Emits every schedule serially (fast), then replays them on N parallel gem5
workers, longest-first.  Appends one CSV row per finished point so partial
progress is always readable.
"""
import csv, json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

SV      = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # repo root, not one host
GEM5    = f"{ROOT}/sim/gem5"
PYBIN   = os.path.expanduser("~/miniconda3/bin/python")
ENV     = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = (os.path.expanduser("~/miniconda3/lib")
                          + f":{GEM5}/ext/dramsim3/DRAMsim3:"
                          + ENV.get("LD_LIBRARY_PATH", ""))
WORKERS = 14

sys.path.insert(0, f"{ROOT}/analytical")
from address_map import AddressMap                        # noqa: E402

def gib_for_frac(tag, frac, db=2):
    m = AddressMap(tag, db, 100.0)
    n = m.g["num_experts"] * len(m.g["moe_layers"])
    return round((m.pinned_bytes + frac * n * m.expert_stride) / 2**30 + 0.05, 2)

# ---- the grid -------------------------------------------------------------
# name, tag, frac, batch, link, dtype, depth, barriers, extras
O, D, P, M = "OLMoE-1B-7B", "DeepSeek-V2-Lite", "Phi-3.5-MoE", "Mixtral-8x7B"
PTS = []
def pt(name, tag, frac, batch=1, link=50.7, dtype="fp16", depth=0,
       barriers=32, gpu="H100", hbm_gbps=708.0, hbm_cfg="HBM3_16Gb_x64_1ch",
       fifo=48, abs_gib=None, bus=4.0, fifo_model=None, t0=None,
       policy="static", phase="decode", warmup=0, fill=False):
    db = 2 if dtype == "fp16" else 1
    PTS.append(dict(name=name, tag=tag, batch=batch,
                    hbm=abs_gib if abs_gib else gib_for_frac(tag, frac, db),
                    link=link, dtype=dtype, depth=depth, barriers=barriers,
                    gpu=gpu, hbm_gbps=hbm_gbps, hbm_cfg=hbm_cfg, fifo=fifo,
                    bus=bus, fifo_model=fifo_model, t0=t0,
                    policy=policy, phase=phase, warmup=warmup,
                    fill=fill))

# G: expert granularity, 4 models, miss-heavy
pt("G-olmoe",    O, .05); pt("G-deepseek", D, .05)
pt("G-phi",      P, .05); pt("G-mixtral",  M, .05)
# C: capacity / miss-rate ladder
pt("C-olmoe-2.5",   O, .025); pt("C-olmoe-7.5",  O, .075)
pt("C-olmoe-10",    O, .10);  pt("C-olmoe-15",   O, .15)
pt("C-mixtral-7.5", M, .075); pt("C-mixtral-10", M, .10)
pt("C-mixtral-15",  M, .15)
pt("C-deepseek-2.5", D, .025); pt("C-phi-2.5", P, .025)
# L: link bandwidth
pt("L-olmoe-pcie4",   O, .05, link=23.5)
pt("L-olmoe-cxl3",    O, .05, link=105.6, fifo=128)
pt("L-mixtral-pcie4", M, .05, link=23.5)
pt("L-mixtral-cxl3",  M, .05, link=105.6, fifo=128)
pt("L-phi-cxl3",      P, .05, link=105.6, fifo=128)
# B: batch union
pt("B-olmoe-b4",   O, .10, batch=4);  pt("B-olmoe-b16", O, .10, batch=16)
pt("B-olmoe-b64",  O, .10, batch=64)
pt("B-mixtral-b4", M, .10, batch=4);  pt("B-mixtral-b16", M, .10, batch=16)
# D: fp8
pt("D-olmoe-fp8",   O, .05, dtype="fp8")
pt("D-mixtral-fp8", M, .05, dtype="fp8")
pt("D-deepseek-fp8", D, .05, dtype="fp8")
# H: HBM3e / H200
pt("H-olmoe-h200",   O, .05, gpu="H200", hbm_gbps=1027.3,
   hbm_cfg="HBM3e_24Gb_x64_1ch")
pt("H-mixtral-h200", M, .05, gpu="H200", hbm_gbps=1027.3,
   hbm_cfg="HBM3e_24Gb_x64_1ch")
# X: long horizon (drift)
pt("X-olmoe-256",    O, .05, barriers=256)
pt("X-olmoe-1024",   O, .05, barriers=1024)
pt("X-deepseek-128", D, .05, barriers=128)
pt("X-mixtral-96",   M, .10, barriers=96)
# FLAGSHIP: the headline H100 operating point, long window
pt("FLAG-mixtral-b16-80g", M, None, batch=16, barriers=128, abs_gib=80.0)
# K: all-hit controls (HBM + compute side only)
pt("K-olmoe-ctrl",   O, .30, barriers=32)
pt("K-mixtral-ctrl", M, .30, barriers=16)

# ---- phase A: emit schedules (serial, fast) -------------------------------
def emit(p):
    sched = f"{SV}/sched/{p['name'].split('@')[0]}.json"
    cmd = [PYBIN, f"{ROOT}/analytical/trace_gen.py", "--tag", p["tag"],
           "--batch", str(p["batch"]), "--hbm-gib", str(p["hbm"]),
           "--policy", p.get("policy", "static"),
           "--phase", p.get("phase", "decode"), "--dtype", p["dtype"],
           "--link-gbps", str(p["link"]), "--gpu", p["gpu"],
           "--hbm-gbps", str(p["hbm_gbps"]),
           "--prefetch-depth", str(p["depth"]),
           "--max-barriers", str(p["barriers"]), "--emit-schedule", sched]
    if p.get("fifo_model"):
        cmd += ["--fifo-entries", str(p["fifo_model"])]
    if p.get("t0") is not None:
        cmd += ["--t0-us", str(p["t0"])]
    if p.get("warmup"):
        cmd += ["--warmup-barriers", str(p["warmup"])]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return None, f"emit failed: {r.stderr[-300:]}"
    s = json.load(open(sched))
    p["sched"] = sched
    p["cxl_gib"] = s["analytic"]["bytes_fetched"] / 2**30
    p["hit"] = s["analytic"]["hit_rate"]
    p["analytic_s"] = s["analytic"]["total_s"]
    p["n_barriers"] = len(s["barriers"])
    return p, None

# ---- phase B: replay (parallel) -------------------------------------------
csv_lock = threading.Lock()
RESULTS = f"{SV}/results.csv"

def replay(p):
    out = f"{SV}/runs/{p['name']}"
    os.makedirs(out, exist_ok=True)
    cmd = [f"{GEM5}/build/NULL/gem5.opt", f"--outdir={out}",
           f"{ROOT}/sim/configs/moe_replay.py", "--schedule", p["sched"],
           "--hbm-config", p["hbm_cfg"], "--fifo", str(p["fifo"]),
           "--bus-ghz", str(p.get("bus", 4.0))]
    if p.get("fill"):
        cmd.append("--fill-writes")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=GEM5, env=ENV)
    wall = time.time() - t0
    m = re.search(r"REPLAY barriers=(\d+) total=([\d.]+) ms\s+"
                  r"analytic=([\d.]+) ms\s+ratio=([\d.]+)", r.stdout)
    row = dict(name=p["name"], tag=p["tag"], batch=p["batch"],
               hbm_gib=p["hbm"], link=p["link"], dtype=p["dtype"],
               depth=p["depth"], barriers=p["n_barriers"],
               policy=p.get("policy","static"), phase=p.get("phase","decode"),
               warmup=p.get("warmup",0), gpu=p.get("gpu","H100"),
               hbm_gbps=p.get("hbm_gbps",708.0), fifo=p.get("fifo",48),
               bus=p.get("bus",4.0), t0=p.get("t0"),
               fill=int(bool(p.get("fill"))),
               hit=round(p["hit"], 4), cxl_gib=round(p["cxl_gib"], 2),
               replay_ms=m.group(2) if m else "",
               analytic_ms=m.group(3) if m else "",
               ratio=m.group(4) if m else "FAIL",
               wall_min=round(wall / 60, 1))
    if not m:
        open(f"{out}/FAILED.log", "w").write(r.stdout[-2000:] + r.stderr[-2000:])
    with csv_lock:
        new = not os.path.exists(RESULTS)
        with open(RESULTS, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new: w.writeheader()
            w.writerow(row)
    print(f"[{time.strftime('%H:%M')}] {p['name']:22s} ratio={row['ratio']}"
          f"  ({row['wall_min']} min)", flush=True)

def main():
    os.makedirs(f"{SV}/sched", exist_ok=True)
    os.makedirs(f"{SV}/runs", exist_ok=True)
    good = []
    for p in PTS:
        q, err = emit(p)
        if err: print(f"SKIP {p['name']}: {err}", flush=True)
        else:
            good.append(q)
            print(f"sched {q['name']:22s} cxl={q['cxl_gib']:7.2f} GiB "
                  f"hit={q['hit']:6.1%} barriers={q['n_barriers']}", flush=True)
    good.sort(key=lambda p: -p["cxl_gib"])          # longest first
    print(f"\n{len(good)} points, {sum(p['cxl_gib'] for p in good):.0f} GiB "
          f"total CXL traffic, {WORKERS} workers\n", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(replay, good))
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
