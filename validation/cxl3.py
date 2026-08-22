#!/usr/bin/env python
"""CXL3-priority validation batch: the 117 GB/s fetch constant across
capacity, batch, dtype and horizon, plus the new Little's-law FIFO cap
(48-entry silicon FIFOs -> 87.8 GB/s analytic) validated against gem5.

All points: exact-ceiling link stage (bus 15.125 GHz, width 8 = 121.0 GB/s),
analytic link 117 (fetch calibration), static placement, current model.
"""
import driver

driver.PTS.clear()
O, M = "OLMoE-1B-7B", "Mixtral-8x7B"
def pt(name, tag, frac, **kw):
    kw.setdefault("link", 117.0); kw.setdefault("fifo", 128)
    kw.setdefault("bus", 15.125)
    driver.pt(name, tag, frac, **kw)

# capacity / burstiness gradient on CXL3
pt("C3-olmoe-2.5",   O, .025); pt("C3-olmoe-7.5", O, .075)
pt("C3-olmoe-15",    O, .15)
pt("C3-mixtral-7.5", M, .075); pt("C3-mixtral-15", M, .15)
# batch unions on CXL3
pt("B3-olmoe-b4",   O, .10, batch=4); pt("B3-olmoe-b16", O, .10, batch=16)
pt("B3-mixtral-b4", M, .10, batch=4)
# dtype on CXL3
pt("D3-olmoe-fp8",   O, .05, dtype="fp8")
pt("D3-mixtral-fp8", M, .05, dtype="fp8")
# silicon 48-entry FIFOs: analytic capped at 87.8 by the new min(), replay
# gets the real 48-entry bridges -- validates the cap mechanism itself
pt("F48-olmoe-5",    O, .05, fifo=48, fifo_model=48)
pt("F48-mixtral-10", M, .10, fifo=48, fifo_model=48)
# long horizon on CXL3
pt("X3-olmoe-256",   O, .05, barriers=256)

driver.RESULTS = f"{driver.SV}/cxl3-results.csv"
driver.WORKERS = 8
driver.main()
