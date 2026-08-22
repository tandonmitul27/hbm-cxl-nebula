#!/usr/bin/env python
"""Close the last two coverage gaps left by the campaign.

  BB  B200        -- 26,880 sweep rows (1/3 of the grid) were pure
                     extrapolation: B200 is the same HBM3e device as H200
                     but 8 stacks instead of 6, and the stack-count axis had
                     never been replayed above 6.
  PF2 prefill     -- n=10 against decode's n=68, and the weakest regime at
                     +/-3.0%. Adds the batch, dtype and link axes that the
                     first prefill batch did not cross.

Emitted at the FINAL constants (HBM3 736.6 / HBM3e 1072.1; links 24.9 /
50.7 / 114.5 with the matching replay bus + FIFO), so these ratios need no
post-hoc link correction -- unlike the earlier batches, whose schedules were
emitted at the pre-calibration link values.
"""
import driver
driver.PTS.clear()
O, D, P, M = "OLMoE-1B-7B", "DeepSeek-V2-Lite", "Phi-3.5-MoE", "Mixtral-8x7B"
HBM3E = dict(hbm_gbps=1072.1, hbm_cfg="HBM3e_24Gb_x64_1ch")


def pt(n, tag, frac, **kw):
    kw.setdefault("t0", 0.0)                 # validation convention
    kw.setdefault("hbm_gbps", 736.6)
    lk = kw.get("link", 50.7)
    if lk == 114.5:                          # CXL 3.0 x16
        kw.setdefault("fifo", 128); kw.setdefault("bus", 15.125)
    elif lk == 24.9:                         # PCIe 4.0 x16
        kw.setdefault("bus", 6.5)
    driver.pt(n, tag, frac, **kw)


# ---- BB: B200 / 8 stacks of HBM3e ----------------------------------------
for tag, ab in [(O, 'olmoe'), (D, 'deepseek'), (P, 'phi'), (M, 'mixtral')]:
    pt(f"BB-{ab}", tag, .05, gpu="B200", **HBM3E)
pt("BB-mixtral-b16",  M, .10, batch=16, gpu="B200", **HBM3E)
pt("BB-olmoe-b16",    O, .10, batch=16, gpu="B200", **HBM3E)
pt("BB-mixtral-fp8",  M, .05, dtype="fp8", gpu="B200", **HBM3E)
pt("BB-olmoe-cxl3",   O, .05, link=114.5, gpu="B200", **HBM3E)
pt("BB-mixtral-cxl3", M, .05, link=114.5, gpu="B200", **HBM3E)
pt("BB-olmoe-pcie4",  O, .05, link=24.9,  gpu="B200", **HBM3E)
pt("BB-phi-hr",       P, .50, barriers=64, gpu="B200", **HBM3E)
pt("BB-olmoe-x256",   O, .05, barriers=256, gpu="B200", **HBM3E)

# ---- PF2: prefill across the axes the first prefill batch did not cross ---
pt("PF2-olmoe-b4",     O, .10, phase="prefill", batch=4,  barriers=8)
pt("PF2-olmoe-b16",    O, .10, phase="prefill", batch=16, barriers=8)
pt("PF2-mixtral-b4",   M, .10, phase="prefill", batch=4,  barriers=8)
pt("PF2-deepseek-b16", D, .10, phase="prefill", batch=16, barriers=8)
pt("PF2-olmoe-fp8",    O, .10, phase="prefill", dtype="fp8", barriers=8)
pt("PF2-mixtral-fp8",  M, .10, phase="prefill", dtype="fp8", barriers=8)
pt("PF2-olmoe-pcie4",  O, .10, phase="prefill", link=24.9, barriers=8)
pt("PF2-phi-pcie4",    P, .10, phase="prefill", link=24.9, barriers=8)
pt("PF2-deepseek-cxl3", D, .25, phase="prefill", link=114.5, barriers=8)
pt("PF2-phi-cxl3",     P, .25, phase="prefill", link=114.5, barriers=8)
pt("PF2-olmoe-h200",   O, .10, phase="prefill", gpu="H200", barriers=8, **HBM3E)
pt("PF2-mixtral-50",   M, .50, phase="prefill", barriers=16)
pt("PF2-olmoe-long",   O, .25, phase="prefill", barriers=32)

driver.RESULTS = f"{driver.SV}/fill-results.csv"
driver.WORKERS = 20
driver.main()
