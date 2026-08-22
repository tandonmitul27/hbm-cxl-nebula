#!/usr/bin/env python
"""Consolidate the campaign into ONE self-contained per-point table.

The analysis previously had to be run from the campaign working directory:
it re-derived each point's analytic prediction from a schedule JSON, so the
published script could not actually reproduce the published numbers. This
freezes everything the analysis needs -- the gem5 measurement, the analytic
prediction recomputed at the FINAL constants, and the harness-artifact
regressor -- into docs/replay_points.csv, and ships the schedules alongside
so any point can still be re-replayed from scratch.
"""
import csv, json, os, re, sys

SV = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repo root, not one host
OUT = f"{ROOT}/docs/replay_points.csv"

FIFO_CAP = 8 * 48 * 64 / 242e-9 / 1e9
# Points whose replay ran against a link stage that was a harness ceiling
# rather than the link (see MODEL.md 5b, "Harness and constant corrections").
# They are excluded, not corrected: the measurement is of the wrong thing.
BAD = {'L-olmoe-cxl3', 'L-mixtral-cxl3', 'L-phi-cxl3', 'A-deepseek-cxl3',
       'L-olmoe-pcie4', 'L-mixtral-pcie4', 'A-phi-pcie4'}
CSVS = ['results.csv', 'aux-results.csv', 'cxl3-results.csv',
        'gap-results.csv', 'fill-results.csv']


def recur(s, E):
    """The t0-free recurrence, in the form the replay gates on."""
    hbm = s['hbm_gbps_per_stack'] * 1e9 * s['stacks']
    link = E * 1e9
    now = lf = 0.0
    for b in s['barriers']:
        nb = b['cxl_bytes']; fl = b['t_compute_floor_s']
        arrive = (max(now, lf) + 280e-9 + nb / link + nb / hbm) if nb else now
        if nb:
            lf = arrive
        tr = (b['hbm_bytes'] - nb) / hbm; tm = nb / hbm
        now = max(max(now + tr, arrive) + tm, now + fl)
    return now


# Schedules emitted before trace_gen.py recorded the GPU carry only the
# memory system it was configured from, which identifies it uniquely.
BY_MEMSYS = {(5, "HBM3"): "H100", (6, "HBM3e"): "H200", (8, "HBM3e"): "B200"}


def gpu_of(s):
    if s.get("gpu"):
        return s["gpu"], s.get("hbm_gen")
    gen = "HBM3e" if s["hbm_gbps_per_stack"] > 900 else "HBM3"
    g = BY_MEMSYS.get((s["stacks"], gen))
    if g is None:
        raise SystemExit(f"unknown memory system: {s['stacks']} x {gen}")
    return g, gen


def main():
    reps = {}
    for f in CSVS:
        p = f'{SV}/{f}'
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r['ratio'] != 'FAIL':
                reps[r['name']] = (float(r['replay_ms']) / 1e3,
                                   r.get('phase', 'decode'),
                                   r.get('policy', 'static'))
    # F2/F3: the two link-constant refits, whose totals live in run logs
    for d in ['F2', 'F3']:
        for sub in sorted(os.listdir(f'{SV}/runs')):
            if sub.startswith(d + '-'):
                lg = f'{SV}/runs/{sub}/run.log'
                if os.path.exists(lg):
                    m = re.search(r'total=([\d.]+) ms', open(lg).read())
                    if m:
                        reps[sub] = (float(m.group(1)) / 1e3, 'decode', 'static')

    rows = []
    for n in sorted(reps):
        rep, ph, pol = reps[n]
        if n in BAD or n.startswith('V-'):
            continue
        base = n[3:] if n.startswith(('F2-', 'F3-')) else n
        sp = f'{SV}/sched/{base}.json'
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp))
        E = s['link_gbps_effective']
        # schedules emitted before the link refits carry the old constants;
        # the replay ran at the corrected stage, so score against the
        # corrected value. Points emitted after the refit already match.
        if abs(E - 105.6) < .1 or abs(E - 117.0) < .1:
            E = 114.5
        if abs(E - 23.5) < .1 or n.startswith('F3-'):
            E = 24.9
        link = E                     # the LINK the point is testing
        if n.startswith('F48'):      # ... which a shallow device FIFO caps
            E = min(E, FIFO_CAP)     # below its own rate (Little's law)
        a = recur(s, E)
        if a <= 0:
            continue
        gpu, gen = gpu_of(s)
        rows.append(dict(
            name=n, phase=ph, policy=pol, gpu=gpu,
            tag=s['tag'], batch=s['batch'], dtype=s['dtype'],
            stacks=s['stacks'], hbm_gen=gen,
            link_gbps=round(link, 1), eff_link_gbps=round(E, 1),
            barriers=len(s['barriers']),
            hit_rate=round(s['analytic']['hit_rate'], 4),
            replay_s=f"{rep:.9g}", analytic_s=f"{a:.9g}",
            ratio=f"{rep / a:.6f}",
            # replay quantum as a fraction of a barrier: the measured harness
            # sampling artifact's regressor (MODEL.md 5b)
            q_over_bar=f"{4000e-9 / (a / len(s['barriers'])):.6f}"))

    os.makedirs(f"{ROOT}/docs", exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} points -> {OUT}")
    by = {}
    for r in rows:
        by[r['gpu']] = by.get(r['gpu'], 0) + 1
    print("  by gpu:", by)
    print("  prefill:", sum(1 for r in rows if r['phase'] == 'prefill'))


if __name__ == "__main__":
    main()
