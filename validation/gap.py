#!/usr/bin/env python
"""Close the three validation gaps the static campaign left open.

  PF  prefill        -- n=0 today; structurally the most different regime
                        (compute-bound, batch-union approaching every expert)
                        and half the sweep's rows.
  DY  dynamic policy -- results/README.md HEADLINE numbers quote lru_layer,
                        but the 53-point campaign was 100% static. Old
                        dynamic points (s1/s4/s5) predate the recalibrated
                        link constants.
  HR  high residency -- validation clustered at 2.5-15% capacity because
                        short windows saturate; the QUOTED regime is 91-100%.
                        Long windows make high-residency points meaningful.

All t0-free (validation convention), current constants, exact link ceilings.
"""
import driver
driver.PTS.clear()
O,D,P,M = "OLMoE-1B-7B","DeepSeek-V2-Lite","Phi-3.5-MoE","Mixtral-8x7B"
def pt(n,tag,frac,**kw):
    kw.setdefault("t0",0.0)
    if kw.get("link")==117.0: kw.setdefault("fifo",128); kw.setdefault("bus",15.125)
    elif kw.get("link")==24.9: kw.setdefault("bus",6.5)
    driver.pt(n,tag,frac,**kw)

# ---- PF: prefill, all 4 models, two capacities, both links ----------------
for tag,ab in [(O,'olmoe'),(D,'deepseek'),(P,'phi'),(M,'mixtral')]:
    pt(f"PF-{ab}-25", tag, .25, phase="prefill", barriers=8)
    pt(f"PF-{ab}-10", tag, .10, phase="prefill", barriers=8)
pt("PF-olmoe-cxl3",   O, .25, phase="prefill", barriers=8, link=117.0)
pt("PF-mixtral-cxl3", M, .25, phase="prefill", barriers=8, link=117.0)

# ---- DY: dynamic policies at the NEW constants ----------------------------
for pol in ["lru_layer","lru","none"]:
    pt(f"DY-olmoe-{pol}",   O, .10, policy=pol, barriers=64, warmup=16 if pol!="none" else 0)
    pt(f"DY-mixtral-{pol}", M, .10, policy=pol, barriers=32, warmup=8  if pol!="none" else 0)
pt("DY-olmoe-oracle",   O, .10, policy="oracle", barriers=64)
pt("DY-deepseek-lru_layer", D, .10, policy="lru_layer", barriers=64, warmup=16)
pt("DY-phi-lru_layer",      P, .10, policy="lru_layer", barriers=32, warmup=8)
pt("DY-olmoe-lru_layer-cxl3", O, .10, policy="lru_layer", barriers=64,
   warmup=16, link=117.0)

# ---- HR: high residency, long windows (the quoted regime) -----------------
pt("HR-mixtral-static-80",  M, None, abs_gib=80.0, barriers=128)
pt("HR-mixtral-lru-80",     M, None, abs_gib=80.0, barriers=128,
   policy="lru_layer", warmup=32)
pt("HR-olmoe-90",  O, .90, barriers=256)
pt("HR-phi-75",    P, .75, barriers=128)
pt("HR-deepseek-90", D, .90, barriers=128)

driver.RESULTS = f"{driver.SV}/gap-results.csv"
driver.WORKERS = 20
driver.main()
