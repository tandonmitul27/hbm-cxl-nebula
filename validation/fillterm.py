#!/usr/bin/env python
"""Is the analytic fill term in the wrong place?

The replay says a cache-fill write costs ZERO time: FW-olmoe-h100 with and
without --fill-writes simulates to the same 0.007804 s, while gem5's counters
confirm 4.57 MB actually written per sampled channel. The fill overlaps the
transfer that produces it -- each 64 B lands in HBM as it arrives -- so it is
hidden under nb/link, which is always >> nb/hbm.

The recurrence instead charges it SERIALLY, after the transfer:

    arrive = start + xfer_lat + nb/link + nb/hbm       <- the last term

That spurious term is worth link/hbm of the barrier, so it depresses the
ratio more when HBM is slow. Exactly the stack-count trend in the residual:
H100 (3683 GB/s) 0.9907, H200 (6433) 1.0038, B200 (8577) 1.0084.

Rescore all points under three variants and see which flattens it. Pure
arithmetic on the frozen schedules -- no new gem5 runs.
"""
import csv, json, os, statistics as st

SV = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repo root, not one host


def recur(s, E, mode):
    hbm = s['hbm_gbps_per_stack'] * 1e9 * s['stacks']
    link = E * 1e9
    now = lf = 0.0
    for b in s['barriers']:
        nb = b['cxl_bytes']; fl = b['t_compute_floor_s']
        if nb:
            # 'serial'   : fill charged after the transfer (today)
            # 'hidden'   : fill overlaps the transfer, costs nothing
            # 'bandwidth': hidden, but the write trickle steals HBM read
            #              bandwidth for the duration of the transfer
            fill = nb / hbm if mode == 'serial' else 0.0
            arrive = max(now, lf) + 280e-9 + nb / link + fill
            lf = arrive
        else:
            arrive = now
        rd = hbm - link if (mode == 'bandwidth' and nb) else hbm
        tr = (b['hbm_bytes'] - nb) / rd
        tm = nb / rd
        now = max(max(now + tr, arrive) + tm, now + fl)
    return now


def main():
    pts = list(csv.DictReader(open(f"{ROOT}/docs/replay_points.csv")))
    out = {}
    for p in pts:
        sp = f"{SV}/sched/{p['name'].split('@')[0]}.json"
        base = p['name']
        if base.startswith(('F2-', 'F3-')):
            sp = f"{SV}/sched/{base[3:]}.json"
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp))
        rep = float(p['replay_s'])
        E = float(p['eff_link_gbps'])
        key = (p['hbm_gen'], int(p['stacks']))
        for mode in ('serial', 'hidden', 'bandwidth'):
            a = recur(s, E, mode)
            out.setdefault(mode, {}).setdefault(key, []).append(rep / a)

    print(f"{'variant':11s} {'memory system':22s} {'n':>4s} {'mean':>8s} "
          f"{'sd':>7s}")
    for mode in ('serial', 'hidden', 'bandwidth'):
        g = out[mode]
        allv = [x for v in g.values() for x in v]
        for k in sorted(g, key=lambda k: k[1]):
            v = g[k]
            print(f"{mode:11s} {k[0]+' x'+str(k[1]):22s} {len(v):4d} "
                  f"{st.mean(v):8.4f} {st.stdev(v):7.4f}")
        means = [st.mean(v) for v in g.values()]
        print(f"{'':11s} {'-> spread across systems':22s} {len(allv):4d} "
              f"{max(means)-min(means):8.4f} {st.stdev(allv):7.4f}   "
              f"(overall mean {st.mean(allv):.4f})\n")


if __name__ == "__main__":
    main()
