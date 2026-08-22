#!/usr/bin/env python
"""Supplementary static-validation points on the 6 spare workers.

Fills the axis gaps the main grid left on the middle-granularity models
(DeepSeek 16.5 MiB, Phi 150 MiB): link, batch and dtype coverage there was
OLMoE/Mixtral-only.  Runs beside the main driver -- distinct point names, so
distinct schedule files and outdirs; results go to aux-results.csv.
"""
import driver

driver.PTS.clear()
O, D, P, M = "OLMoE-1B-7B", "DeepSeek-V2-Lite", "Phi-3.5-MoE", "Mixtral-8x7B"
pt = driver.pt
pt("A-deepseek-cxl3",  D, .05, link=105.6, fifo=128)   # link axis, fine experts
pt("A-phi-pcie4",      P, .05, link=23.5)              # link axis, medium experts
pt("A-deepseek-b16",   D, .10, batch=16)               # batch axis, 2nd family
pt("A-phi-b4",         P, .05, batch=4)                # batch axis, medium experts
pt("A-phi-fp8",        P, .05, dtype="fp8")            # dtype axis, medium experts
pt("A-deepseek-7.5",   D, .075)                        # capacity ladder, DeepSeek

driver.RESULTS = f"{driver.SV}/aux-results.csv"
driver.WORKERS = 6
driver.main()
