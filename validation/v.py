#!/usr/bin/env python
"""t0-free re-runs of the short CXL3 points: validation isolates the memory
recurrence (t0 is exact serial arithmetic on both sides, nothing to validate;
inside the replay's max() it hides, outside the analytic's it pays -- so it
must be zero in cross-checks)."""
import driver
driver.PTS.clear()
O = "OLMoE-1B-7B"
def pt(name, tag, frac, **kw):
    kw.setdefault("link", 117.0); kw.setdefault("fifo", 128)
    kw.setdefault("bus", 15.125); kw.setdefault("t0", 0.0)
    driver.pt(name, tag, frac, **kw)
pt("V-C3-olmoe-2.5", O, .025); pt("V-C3-olmoe-7.5", O, .075)
pt("V-C3-olmoe-15",  O, .15)
pt("V-D3-olmoe-fp8", O, .05, dtype="fp8")
pt("V-F48-olmoe-5",  O, .05, fifo=48, fifo_model=48)
driver.RESULTS = f"{driver.SV}/v-results.csv"
driver.WORKERS = 5
driver.main()
