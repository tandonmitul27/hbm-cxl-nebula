#!/usr/bin/env python
"""Composed-energy results as the table MODEL.md carries."""
import glob, json, os, statistics as st, sys

SV = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(f"{SV}/energy/*.result.json")):
    r = json.load(open(f))
    if "hbm_gen" not in r or r["point"] == "SMOKE":
        continue                       # SMOKE predates the bucket fix
    r["dyn"] = r["gem5_dyn_j"] / r["model_dyn_j"]
    r["bg"] = r["gem5_stb_j"] / r["model_bg_j"]
    rows.append(r)

for fill in (False, True):
    v = [r for r in rows if r["fill_writes"] == fill]
    if not v:
        continue
    print(f"\n--- fill_writes={fill}  (n={len(v)}) ---")
    print(f"{'point':22s} {'gen':6s} {'u_hbm':>6s} {'dyn':>7s} {'bg':>7s} "
          f"{'total':>7s}")
    for r in sorted(v, key=lambda x: x["point"]):
        print(f"{r['point']:22s} {r['hbm_gen']:6s} {r['u_hbm']:6.3f} "
              f"{r['dyn']:7.3f} {r['bg']:7.3f} {r['ratio']:7.3f}")
    for lbl, key in (("dynamic", "dyn"), ("background", "bg"),
                     ("composite", "ratio")):
        x = [r[key] for r in v]
        print(f"  {lbl:11s} {min(x):.3f} - {max(x):.3f}   mean {st.mean(x):.3f}")

# split by duty cycle: the regime where the saturated endpoint used to bill wrong
low = [r for r in rows if r["u_hbm"] < 0.5]
hi = [r for r in rows if r["u_hbm"] >= 0.5]
print(f"\nbackground term, stall-dominated (u_hbm<0.5, n={len(low)}): "
      f"{st.mean(r['bg'] for r in low):.3f}")
print(f"background term, saturated      (u_hbm>=0.5, n={len(hi)}): "
      f"{st.mean(r['bg'] for r in hi):.3f}")
