#!/usr/bin/env python
"""Composed-energy validation: replay a real MoE decode window on gem5 with
per-instance DRAMsim3 energy output, sum it, and compare against what
analytical/energy_model.py predicts for the same byte counts.

Why this exists: energy was validated PER DEVICE (HBM pJ/bit vs O'Connor,
DDR5 vs the device class) but the COMPOSED figure -- HBM reads + HBM fills +
expander reads + background power over a whole run -- had never been checked
against gem5. This closes that gap.

Scaling. The replay models ONE stack's worth of traffic divided across
`--hbm-channels` sampled channels (per-channel share = hbm_bytes / stacks /
16), and the full 8-channel expander. So:

    HBM energy  x (16 / n_sampled) x stacks
    CXL energy  x 1

Units: DRAMsim3 emits V*mA*CYCLES, not pJ -- multiply by each config's tCK
(sim/measure_energy.py, CALIBRATION.md "Units").
"""
import json, os, re, subprocess, sys

SV = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repo root, not one host
GEM5 = f"{ROOT}/sim/gem5"
DRAMSIM = f"{GEM5}/ext/dramsim3/DRAMsim3"
PY = os.path.expanduser("~/miniconda3/bin/python")
sys.path.insert(0, f"{ROOT}/analytical")

# Bucketing must match the MODEL's split, not DRAMsim3's. The model bills
# refresh as BACKGROUND (it is a duty the device performs whether or not the
# host asks for anything), so ref_energy/refb_energy belong on the background
# side here. Putting them with the dynamic terms -- DRAMsim3's own grouping --
# makes the per-component ratios meaningless while leaving the total right.
DYNAMIC_KEYS = ["act_energy", "read_energy", "write_energy"]
BACKGROUND_KEYS = ["ref_energy", "refb_energy",
                   "act_stb_energy", "pre_stb_energy", "sref_energy"]


def tck(cfg):
    txt = open(f"{DRAMSIM}/configs/{cfg}.ini").read()
    return float(re.search(r"^tCK = ([\d.]+)", txt, re.M).group(1))


def _flatten(v):
    """Standby energies are per-RANK dicts; dynamic ones are scalars.
    Same helper measure_energy.py uses -- reused rather than reimplemented."""
    if isinstance(v, dict):
        return sum(float(x) for x in v.values())
    return float(v)


def read_json_energy(d, cfg):
    """pJ from one DRAMsim3 instance dir (V*mA*cycles -> pJ via tCK)."""
    f = os.path.join(d, "dramsim3.json")
    if not os.path.exists(f):
        return None
    raw = json.load(open(f))
    chans = list(raw.values()) if all(k.isdigit() for k in raw) else [raw]
    t = tck(cfg)
    dyn = sum(_flatten(c.get(k, 0)) for c in chans for k in DYNAMIC_KEYS) * t
    bg = sum(_flatten(c.get(k, 0)) for c in chans for k in BACKGROUND_KEYS) * t
    return dyn, bg


HBM_CFG = {"HBM3": "HBM3_16Gb_x64_1ch", "HBM3e": "HBM3e_24Gb_x64_1ch"}


def main(point, fill=False, bus=4.0, fifo=48):
    sched_p = f"{SV}/sched/{point}.json"
    sched = json.load(open(sched_p))
    gen = sched.get("hbm_gen", "HBM3")
    suffix = "-fill" if fill else ""
    edir = f"{SV}/energy/{point}{suffix}"
    os.makedirs(edir, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.expanduser("~/miniconda3/lib") + ":"
                              + DRAMSIM + ":" + env.get("LD_LIBRARY_PATH", ""))
    n_hbm = 4
    cmd = [f"{GEM5}/build/NULL/gem5.opt",
           f"--outdir={SV}/runs/E-{point}{suffix}",
           f"{ROOT}/sim/configs/moe_replay.py", "--schedule", sched_p,
           "--hbm-config", HBM_CFG[gen], "--hbm-channels", str(n_hbm),
           "--bus-ghz", str(bus), "--fifo", str(fifo),
           "--energy-dir", edir] + (["--fill-writes"] if fill else [])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=GEM5, env=env)
    m = re.search(r'\{"replay_total_s".*\}', r.stdout)
    if not m:
        print("REPLAY FAILED\n" + r.stdout[-1500:] + r.stderr[-1500:]); return
    out = json.loads(m.group(0))
    stacks = out["hbm_scale"]

    hbm_dyn = hbm_stb = cxl_dyn = cxl_stb = 0.0
    for e in out["energy_dirs"]:
        got = read_json_energy(e["dir"], e["config"])
        if got is None:
            continue
        d, s = got
        if e["tag"].startswith("hbm"):
            hbm_dyn += d; hbm_stb += s
        else:
            cxl_dyn += d; cxl_stb += s
    # scale sampled HBM channels up to the real stack count
    k = (16.0 / n_hbm) * stacks
    hbm_dyn *= k; hbm_stb *= k

    T = out["replay_total_s"]
    a = sched["analytic"]
    from energy_model import EnergyModel
    em = EnergyModel(gen, n_hbm_channels=16 * stacks, n_ddr5_subchannels=8,
                     e_link_ctrl=0.0)          # link term excluded: not in gem5
    # Background is state-resolved: charge active standby only for the
    # fraction of the window each device is actually streaming. Duty cycles
    # come from the recurrence that produced this schedule, so the model is
    # scored exactly as the sweep uses it.
    fill_bytes = a["bytes_fetched"] if fill else 0.0
    pred = em.energy(a["bytes_hbm_read"], fill_bytes,
                     a["bytes_fetched"], T,
                     u_hbm=a.get("u_hbm"), u_ddr5=a.get("u_ddr5"))

    gem5_dram = (hbm_dyn + hbm_stb + cxl_dyn + cxl_stb) * 1e-12
    pred_dram = (pred["e_hbm_read_j"] + pred["e_hbm_fill_j"]
                 + pred["e_cxl_read_j"] + pred["e_background_j"])
    print(f"\n=== composed energy: {point} ===")
    print(f"  window {T*1e3:.2f} ms   hbm_read {a['bytes_hbm_read']/2**30:.2f} GiB"
          f"   cxl {a['bytes_fetched']/2**30:.2f} GiB")
    print(f"  gem5   dynamic {(hbm_dyn+cxl_dyn)*1e-12:9.4f} J   "
          f"background {(hbm_stb+cxl_stb)*1e-12:9.4f} J   total {gem5_dram:9.4f} J")
    print(f"  model  dyn+fill+cxl {(pred['e_hbm_read_j']+pred['e_hbm_fill_j']+pred['e_cxl_read_j']):9.4f} J   "
          f"background {pred['e_background_j']:9.4f} J   total {pred_dram:9.4f} J")
    print(f"  {gen}  fill_writes={fill}  u_hbm={a.get('u_hbm'):.3f} "
          f"u_ddr5={a.get('u_ddr5'):.3f}")
    print(f"  RATIO gem5/model = {gem5_dram/max(1e-12,pred_dram):.3f}")
    json.dump({"point": point, "hbm_gen": gen, "fill_writes": fill,
               "gem5_j": gem5_dram, "model_j": pred_dram,
               "gem5_dyn_j": (hbm_dyn+cxl_dyn)*1e-12,
               "gem5_stb_j": (hbm_stb+cxl_stb)*1e-12,
               "model_dyn_j": (pred["e_hbm_read_j"] + pred["e_hbm_fill_j"]
                               + pred["e_cxl_read_j"]),
               "model_bg_j": pred["e_background_j"],
               "u_hbm": a.get("u_hbm"), "u_ddr5": a.get("u_ddr5"),
               "ratio": gem5_dram/max(1e-12, pred_dram), "window_s": T},
              open(f"{SV}/energy/{point}{suffix}.result.json", "w"), indent=1)


if __name__ == "__main__":
    fill = "--fill" in sys.argv
    bus, fifo = 4.0, 48
    if "--cxl3" in sys.argv:
        bus, fifo = 15.125, 128
    elif "--pcie4" in sys.argv:
        bus = 6.5
    for p in [x for x in sys.argv[1:] if not x.startswith("-")]:
        main(p, fill=fill, bus=bus, fifo=fifo)
