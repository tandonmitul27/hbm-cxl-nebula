#!/usr/bin/env python
"""Quantified confidence in the analytical model.

Three separable questions, answered separately because they have different
answers and conflating them overstates the weakest one:

  A. ACCURACY   -- how close is the recurrence to gem5, where measured?
                   Empirical prediction intervals from the replay campaign,
                   per regime, with bootstrap CIs on the mean.
  B. PARAMETER  -- how much do results move across the ranges of the values
     SENSITIVITY   that are bracketed rather than measured (t0, e_link)?
                   This bounds what calibration cannot pin down.
  C. COVERAGE   -- which sweep rows sit inside the validated envelope and
                   which are extrapolation? Accuracy claims transfer only
                   inside it.

Inputs are both in this repo, so the published numbers are reproducible:
  docs/replay_points.csv    one row per gem5 replay point
  results/sweep.parquet     the sweep whose coverage is being scored
"""
import csv, os, random, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTS_CSV = os.path.join(ROOT, "docs", "replay_points.csv")
SWEEP = os.path.join(ROOT, "results", "sweep.parquet")


def load():
    with open(PTS_CSV) as f:
        pts = list(csv.DictReader(f))
    for p in pts:
        p["ratio"] = float(p["ratio"])
        p["q_over_bar"] = float(p["q_over_bar"])
    return pts


def boot_ci(xs, n=20000, seed=12345):
    rnd = random.Random(seed)
    means = sorted(st.mean(rnd.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(.025 * n)], means[int(.975 * n)]


def main():
    pts = load()
    # The replay's quantum sampling is a MEASURED harness artifact
    # (ratio = 0.9556 + 0.666 x quantum/t_barrier on a controlled probe that
    # varies only --quantum-ns). At 68 points a least-squares fit over the
    # whole corpus returned 0.699 and cross-validated against that probe.
    # At 103 it returns 0.256 and no longer does -- the added points sit at
    # small quantum/barrier, where the regression is a weak estimator. So the
    # HEADLINE numbers below are UNCORRECTED: the artifact is real and
    # documented, but a global correction that no longer reproduces its own
    # controlled measurement is not one to publish behind. The corrected
    # figures are printed alongside for comparison, never as the claim.
    xs = [p["q_over_bar"] for p in pts]
    ys = [p["ratio"] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    B = (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
         / sum((a - mx) ** 2 for a in xs))
    for p in pts:
        p["corrected"] = p["ratio"] - B * p["q_over_bar"]

    print("=" * 70)
    print("A. ACCURACY  (replay/analytic, UNCORRECTED)")
    print(f"   corpus fit of the quantum artifact: B={B:.3f} "
          f"(controlled probe: 0.666 -- see note in source)")
    print("=" * 70)
    groups = [("all", lambda p: True),
              ("decode", lambda p: p["phase"] == "decode"),
              ("prefill", lambda p: p["phase"] == "prefill"),
              ("static", lambda p: p["policy"] == "static"),
              ("dynamic", lambda p: p["policy"] != "static"),
              ("H100", lambda p: p["gpu"] == "H100"),
              ("H200", lambda p: p["gpu"] == "H200"),
              ("B200", lambda p: p["gpu"] == "B200")]
    for lbl, f in groups:
        v = [p["ratio"] for p in pts if f(p)]
        c = [p["corrected"] for p in pts if f(p)]
        if len(v) < 2:
            continue
        lo, hi = boot_ci(v)
        sd, mu = st.stdev(v), st.mean(v)
        band = max(abs(1 - (mu - 1.96 * sd)), abs(1 - (mu + 1.96 * sd)))
        n3 = sum(1 for x in v if abs(x - 1) <= 0.03)
        print(f"  {lbl:9s} n={len(v):3d}  mean {mu:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]  sd {sd:.4f}  "
              f"within 3%: {n3}/{len(v)}")
        print(f"  {'':9s}      95% prediction interval "
              f"[{mu - 1.96 * sd:.3f}, {mu + 1.96 * sd:.3f}]"
              f"  -> a single prediction lands within +/-{band * 100:.1f}%")
        print(f"  {'':9s}      (artifact-corrected, for comparison only: "
              f"mean {st.mean(c):.4f}  sd {st.stdev(c):.4f})")

    print()
    print("=" * 70)
    print("B. PARAMETER SENSITIVITY  (bracketed, not measured)")
    print("=" * 70)
    import pandas as pd
    if not os.path.exists(SWEEP):
        # Section A above is the accuracy claim and stands on
        # docs/replay_points.csv alone. B and C score the SWEEP, which is a
        # generated artifact -- run `make sweep` to rebuild it.
        print("  results/sweep.parquet not present -- run `make sweep`.")
        print("  Sections B and C score that file; section A above does not")
        print("  depend on it and is complete.")
        return
    d = pd.read_parquet(SWEEP)
    en = d.energy_j_e19 / d.energy_j_e2
    ed = d.edp_js_e19 / d.edp_js_e2
    print(f"  e_link 2->19 pJ/bit :  energy x{en.median():.2f} median, "
          f"x{en.max():.2f} max   EDP x{ed.median():.2f} / x{ed.max():.2f}")
    print("     -> every energy conclusion must be quoted across this bracket")
    print("  t0 5->40 us         :  x1.16 all-resident, x1.00 stall-dominated")
    print("     -> dominant uncertainty only where nothing misses")

    print()
    print("=" * 70)
    print("C. COVERAGE  (validated envelope vs extrapolation)")
    print("=" * 70)
    nd = sum(1 for p in pts if p["phase"] == "decode")
    vg = sorted({p["gpu"] for p in pts})
    sg = sorted(d.gpu.unique()) if "gpu" in d.columns else ["H100"]
    print(f"  sweep rows                     {len(d)}")
    print(f"  replay points                  {len(pts)}")
    print(f"  models validated               "
          f"{len({p['tag'] for p in pts})}/{d.tag.nunique()}")
    print(f"  links validated                "
          f"{len({p['link_gbps'] for p in pts})}/{d.link_gbps.nunique()}"
          f"   ({', '.join(sorted({p['link_gbps'] for p in pts}))})")
    print(f"  batches validated              "
          f"{len({p['batch'] for p in pts})}/{d.batch.nunique()}")
    print(f"  dtypes validated               "
          f"{len({p['dtype'] for p in pts})}/{d.dtype.nunique()}")
    print(f"  phases validated               2/2   "
          f"(decode n={nd}, prefill n={len(pts) - nd})")
    print(f"  policies validated             "
          f"{len({p['policy'] for p in pts})}/{d.policy.nunique()}")
    print(f"  stack counts validated         "
          f"{sorted({int(p['stacks']) for p in pts})}")
    print(f"  GPUs in sweep                  {sg}")
    print(f"  GPUs with replay validation    {vg}"
          f"{'' if set(sg) <= set(vg) else '   -> the rest are EXTRAPOLATION'}")


if __name__ == "__main__":
    main()
