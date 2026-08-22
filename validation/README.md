# Replay-campaign harness

The drivers that produced `docs/replay_points.csv` (103 gem5 replay points)
and the composed-energy results in MODEL.md §5b.

These are *campaign* scripts, not part of the model. They emit schedules with
`analytical/trace_gen.py --emit-schedule`, replay them on
`sim/configs/moe_replay.py`, and record ratios. They expect a working
directory holding `sched/`, `runs/` and the per-batch result CSVs; `driver.SV`
points at wherever the script itself lives.

| script | what it produced |
|---|---|
| `driver.py` | the base grid + the `pt()` / `emit()` / `replay()` machinery every other driver reuses |
| `aux.py`, `cxl3.py`, `v.py` | link-constant refits and the t0-conversion audit |
| `gap.py` | prefill, dynamic policies, high residency |
| `fill.py` | B200 (8 stacks) and the second prefill batch |
| `fillw.py` | paired `--fill-writes` on/off comparison |
| `regress.py` | proves the `dramsim3.cc` patch is a no-op for read-only traffic |
| `energy_camp.py`, `energy_val.py`, `energy_table.py` | composed-energy validation |
| `export_points.py` | freezes everything into `docs/replay_points.csv` |
| `fillterm.py`, `xfer_scan.py` | offline rescoring used to test (and reject) explanations for the HBM3e residual |

`export_points.py` is the one that matters for reproducibility: after it runs,
`docs/confidence_analysis.py` needs nothing but the repo.
