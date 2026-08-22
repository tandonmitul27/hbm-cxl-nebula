#!/usr/bin/env python
"""Composed-energy re-validation, second pass.

The first pass (4 points, HBM3 only) predated three model changes:
  * state-resolved background power  P_bg(u) = P_ref + u.P_act_stb + (1-u).P_pre_stb
  * corrected HBM3e per-state coefficients (an earlier draft SCALED them from
    HBM3 and mislabelled them measured; refresh was 17% off)
  * --fill-writes in the replay, which removes the one term the old
    comparison could only infer

so its 0.842-0.860 composite is not a score of the model that ships. This
re-runs it across both HBM generations, both dtypes, and both fill settings.
Read-only points stay in: they isolate whether the fill term is the whole of
the old gap.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import driver
import energy_val

SV = driver.SV
O, D, P, M = "OLMoE-1B-7B", "DeepSeek-V2-Lite", "Phi-3.5-MoE", "Mixtral-8x7B"
HBM3E = dict(hbm_gbps=1072.1, hbm_cfg="HBM3e_24Gb_x64_1ch")

driver.PTS.clear()
def pt(n, tag, frac, **kw):
    kw.setdefault("t0", 0.0)
    kw.setdefault("hbm_gbps", 736.6)
    driver.pt(n, tag, frac, **kw)

# HBM3 / H100
pt("EN-olmoe",       O, .10, barriers=32)
pt("EN-deepseek",    D, .10, barriers=32)
pt("EN-mixtral",     M, .10, barriers=16)
pt("EN-olmoe-fp8",   O, .10, dtype="fp8", barriers=32)
pt("EN-olmoe-hr",    O, .50, barriers=64)         # low duty cycle: the case
pt("EN-mixtral-hr",  M, .50, barriers=32)         # saturated bg over-charged
# HBM3e / H200 -- the generation whose coefficients were corrected
pt("EN-olmoe-h200",    O, .10, gpu="H200", barriers=32, **HBM3E)
pt("EN-mixtral-h200",  M, .10, gpu="H200", barriers=16, **HBM3E)
pt("EN-deepseek-h200", D, .10, gpu="H200", barriers=32, **HBM3E)
pt("EN-olmoe-h200-hr", O, .50, gpu="H200", barriers=64, **HBM3E)

for p in driver.PTS:
    q, err = driver.emit(p)
    print(f"{'SKIP ' if err else 'sched'} {p['name']:22s} "
          f"{err or f'''cxl={q['cxl_gib']:6.2f} GiB hit={q['hit']:6.1%}'''}",
          flush=True)

JOBS = [(p["name"], fill) for p in driver.PTS for fill in (False, True)]

def run(job):
    name, fill = job
    t = time.time()
    try:
        energy_val.main(name, fill=fill)
    except Exception as e:                       # one bad point must not
        print(f"ERR {name} fill={fill}: {e}", flush=True)   # sink the batch
    print(f"[{time.strftime('%H:%M')}] done {name} fill={fill} "
          f"({(time.time()-t)/60:.1f} min)", flush=True)

with ThreadPoolExecutor(max_workers=10) as ex:
    list(ex.map(run, JOBS))
print("ALL DONE", flush=True)
