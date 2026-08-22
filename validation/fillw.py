#!/usr/bin/env python
"""Test one hypothesis about the residual's only remaining structure.

Ratios (replay/analytic, CXL2 decode, artifact-corrected) rise monotonically
with aggregate HBM bandwidth and almost nothing else:

    HBM3  x5 stacks (3683 GB/s)  n=45  0.9907  sd 0.0062
    HBM3e x6        (6433)       n= 2  1.0038  sd 0.0004
    HBM3e x8        (8577)       n= 8  1.0084  sd 0.0009

The within-group spread (0.0009 at 8 stacks) says this is systematic, not
noise. The recurrence charges the cache-fill WRITE of every missed expert to
HBM on the arrival path; the read-only replay never issues it. That missing
term is worth nb/hbm per barrier -- it SHRINKS as HBM gets wider, so a
read-only replay should look progressively faster than the analytic as stack
count rises. Exactly the observed sign and ordering.

If that is the cause, running the SAME points with --fill-writes should
collapse the trend. If it does not, the trend is something else and must be
reported as a stack-count-dependent bias rather than explained away.

Matched configurations, each replayed twice (read-only / with fill).
"""
import driver
driver.PTS.clear()
O, D, P = "OLMoE-1B-7B", "DeepSeek-V2-Lite", "Phi-3.5-MoE"
GPU = {"h100": dict(gpu="H100", hbm_gbps=736.6, hbm_cfg="HBM3_16Gb_x64_1ch"),
       "h200": dict(gpu="H200", hbm_gbps=1072.1,
                    hbm_cfg="HBM3e_24Gb_x64_1ch"),
       "b200": dict(gpu="B200", hbm_gbps=1072.1,
                    hbm_cfg="HBM3e_24Gb_x64_1ch")}

for tag, ab in [(O, 'olmoe'), (D, 'deepseek'), (P, 'phi')]:
    for g in ("h100", "h200", "b200"):
        for fill in (False, True):
            # '@' suffix picks a distinct run dir while sharing the schedule
            driver.pt(f"FW-{ab}-{g}" + ("@fill" if fill else ""),
                      tag, .10, barriers=16, t0=0.0, fill=fill, **GPU[g])

driver.RESULTS = f"{driver.SV}/fillw-results.csv"
driver.WORKERS = 18
driver.main()
