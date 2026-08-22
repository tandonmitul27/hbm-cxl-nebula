#!/usr/bin/env python
"""With the fill term hidden (replay-proven), what is left over?

Removing the spurious serial fill halves the cross-system spread but lifts
the overall mean to 1.0123 -- the analytic now UNDER-predicts miss-heavy
barriers by ~1.2%. All-hit points are untouched (nb=0 makes the variants
identical), so whatever is missing lives on the fetch path.

Two candidates, and they are distinguishable:
  * a per-fetch overhead   -> one xfer_lat value should fix ALL links at once
  * a lower link rate      -> each link needs its own correction factor
The 50.7 GB/s CXL2 constant was measured on ONE long synthetic stream; a real
schedule chops the same traffic into many short per-barrier fetches, each
paying startup. That is a physical per-fetch cost, not a fudge -- if it is
the explanation, a single latency will flatten every link together.
"""
import csv, json, os, statistics as st

SV = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repo root, not one host


def recur(s, E, lat):
    hbm = s['hbm_gbps_per_stack'] * 1e9 * s['stacks']
    link = E * 1e9
    now = lf = 0.0
    for b in s['barriers']:
        nb = b['cxl_bytes']; fl = b['t_compute_floor_s']
        arrive = (max(now, lf) + lat + nb / link) if nb else now
        if nb:
            lf = arrive
        tr = (b['hbm_bytes'] - nb) / hbm
        now = max(max(now + tr, arrive) + nb / hbm, now + fl)
    return now


pts = []
for p in csv.DictReader(open(f"{ROOT}/docs/replay_points.csv")):
    n = p['name'].split('@')[0]
    if n.startswith(('F2-', 'F3-')):
        n = n[3:]
    sp = f"{SV}/sched/{n}.json"
    if os.path.exists(sp):
        p['sched'] = json.load(open(sp))
        pts.append(p)

print(f"{'xfer_lat':>9s}  {'mean':>7s} {'sd':>7s}  {'x5':>7s} {'x6':>7s} "
      f"{'x8':>7s} {'spread':>7s}  {'24.9':>7s} {'50.7':>7s} {'114.5':>7s}")
for lat_ns in (280, 500, 800, 1200, 1600, 2000, 2500, 3000):
    rs = []
    for p in pts:
        a = recur(p['sched'], float(p['eff_link_gbps']), lat_ns * 1e-9)
        rs.append((p, float(p['replay_s']) / a))
    v = [r for _, r in rs]
    bys = {}
    byl = {}
    for p, r in rs:
        if float(p['hit_rate']) < 1.0:                 # fetch path only
            byl.setdefault(p['link_gbps'], []).append(r)
        bys.setdefault(int(p['stacks']), []).append(r)
    m = {k: st.mean(x) for k, x in bys.items()}
    ml = {k: st.mean(x) for k, x in byl.items()}
    print(f"{lat_ns:7d}ns  {st.mean(v):7.4f} {st.stdev(v):7.4f}  "
          f"{m.get(5,0):7.4f} {m.get(6,0):7.4f} {m.get(8,0):7.4f} "
          f"{max(m.values())-min(m.values()):7.4f}  "
          f"{ml.get('24.9',0):7.4f} {ml.get('50.7',0):7.4f} "
          f"{ml.get('114.5',0):7.4f}")
