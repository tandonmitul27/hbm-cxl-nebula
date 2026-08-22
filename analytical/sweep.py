"""The full sweep grid: models x batch x capacity x link x policy x depth
x dtype x phase, analytic (gem5-certified inputs; see MODEL.md).

Capacity is swept as a FRACTION of each model's total expert footprint --
{0, 10, 25, 50, 75, 100}% -- so the four granularities land on one
comparable axis; one absolute point per model at the H100's 80 GiB anchors
the story in real hardware.  Pinned weights and the fully-grown KV pool
come off the top before the fraction applies (matching trace_gen).

Every row carries latency, stall, hit rate, bytes, energy at the three
e_link bracket points, and EDP.  Output: results/sweep.parquet.

    python sweep.py                 # full grid, multiprocess
    python sweep.py --tag OLMoE-1B-7B --workers 4    # subset
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# AddressMap is single-sourced in mapping/ -- shared with the gem5
# harnesses and must not fork.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mapping"))

from address_map import AddressMap, GIB                     # noqa: E402
from energy_model import EnergyModel, E_LINK_CTRL_SWEEP     # noqa: E402
from gpu_model import GPUS, attn_quad_flops, peak_flops, t_layer as roofline_t  # noqa: E402
from trace_gen import (DECODE, demand_sequence,             # noqa: E402
                       kv_bytes_per_tok_layer, make_policy, simulate)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

TAGS = ["OLMoE-1B-7B", "DeepSeek-V2-Lite", "Phi-3.5-MoE", "Mixtral-8x7B",
        "Qwen2.5-3B-dense"]   # dense control: 1 "expert"/layer, k=1 --
                              # 100% demand, the no-sparsity baseline
BATCHES = [1, 4, 16, 64]
CAP_FRACS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]   # of expert footprint
H100_GIB = 80.0                                    # absolute anchor point
T0_US = 20.0        # per-barrier launch/dispatch overhead; bracket [5, 40],
                    # see gpu_model.t_layer -- swept in sensitivity runs
LINKS = [24.9, 50.7, 114.5]        # measured effective GB/s for MoE fetch
                                   # streams; CXL3 uses the fetch-stream
                                   # calibration (system_params
                                   # cxl3_x16_121_fetch), not the 105.6
                                   # steady-state figure
POLICIES = ["none", "static", "lru", "lru_layer", "oracle"]
DEPTHS = [0, 1, 2, 4]
DTYPES = ["fp16", "fp8"]
PHASES = ["decode", "prefill"]
# GPU is a SWEEP AXIS, not a constant. Capacity is the dominant variable in
# this study -- Mixtral fp16 (87 GiB) is the only model forced to tier on an
# 80 GiB H100, and that pressure disappears entirely at the H200's 141 GiB --
# so pinning H100 hid the most interesting comparison the model can make.
# Both HBM3 (H100) and HBM3E (H200/B200) per-stack rates are gem5-validated
# (replay ratios 0.992 / 1.004).
GPUS_SWEPT = ["H100", "H200", "B200"]
HBM_GBPS_STACK = {          # measured, sim/README.md (post idle-gap fix;
    "HBM3":  736.6,         # the old 708.0/1027.3 were diluted 4.8% by an
    "HBM3e": 1072.1,        # idle tail inside the measurement window)
}


def run_group(job):
    """All configs for one (tag, batch, phase, dtype) -- the trace and the
    address map load once, then every config replays over them."""
    tag, batch, phase, dtype, gpu_name = job
    GPU = GPUS[gpu_name]
    dtype_bytes = 2 if dtype == "fp16" else 1
    phase_id = DECODE if phase == "decode" else 0
    try:
        seq, run = demand_sequence(tag, batch, phase_id)
    except SystemExit as e:
        return []
    amap = AddressMap(tag, dtype_bytes, 24.0)      # hbm_gib replaced per row
    g = amap.g
    n_layers = len(g["moe_layers"])
    stacks = GPU.stacks
    hbm_bps = HBM_GBPS_STACK[GPU.hbm_gen] * 1e9 * stacks

    base_bytes = amap.att_bytes + amap.shared_bytes
    base_params = base_bytes / dtype_bytes
    P, G = run["prompt_len"], run.get("gen_len", 0)
    tokens = batch if phase == "decode" else batch * P
    kv_bpt = kv_bytes_per_tok_layer(amap.att, dtype_bytes)
    kv_pool = batch * (P + G) * kv_bpt * g["num_layers"]

    def kv_layer_fn(step):
        if phase == "decode":
            return (batch * (P + step) * kv_bpt, batch * kv_bpt)
        return (0.0, batch * P * kv_bpt)

    quad = attn_quad_flops(amap.att, batch, P) if phase == "prefill" else 0.0

    def t_layer_fn(n_experts, step):
        kv_rd, kv_wr = kv_layer_fn(step)
        b = base_bytes + n_experts * amap.expert_bytes + kv_rd + kv_wr
        p = base_params + n_experts * g["expert_params"]
        return roofline_t(b, p, tokens, GPU, dtype_bytes, hbm_bps,
                          t0_s=T0_US * 1e-6, extra_flops=quad)

    def t_compute_fn(n_experts, step):
        # pure compute; t0 is added by simulate() as a serial per-barrier term
        p = base_params + n_experts * g["expert_params"]
        return (2.0 * p * tokens + quad) / peak_flops(GPU, dtype_bytes)

    # capacity basis is the STRIDE footprint (2 MiB-aligned homes), not raw
    # expert bytes -- slots divide by stride, so a bytes basis leaves
    # frac=1.0 ~5% short of the slot count and every pass churns
    expert_total = amap.expert_stride * g["num_experts"] * n_layers
    total_slots = g["num_experts"] * n_layers

    # capacity points: fractions of expert footprint + the 80 GiB anchor
    # the absolute anchor is now the GPU's OWN capacity, not always 80 GiB
    cap_points = [("frac", f) for f in CAP_FRACS] + [("abs", None)]

    rows = []
    for (cap_kind, frac), link, pol_name, depth in itertools.product(
            cap_points, LINKS, POLICIES, DEPTHS):
        if cap_kind == "frac":
            cache_bytes = frac * expert_total
            hbm_gib = (amap.pinned_bytes + kv_pool + cache_bytes) / GIB
        else:
            # the GPU's real capacity: 80 / 141 / 192 GiB
            hbm_gib = GPU.hbm_gib
            cache_bytes = max(0.0, hbm_gib * GIB - amap.pinned_bytes
                              - kv_pool)
            frac = min(1.0, cache_bytes / expert_total)
        slots = int(cache_bytes // amap.expert_stride)
        eff_slots = max(0, slots - depth * g["top_k"])

        pol = make_policy(pol_name, eff_slots, seq, n_layers)
        r = simulate(seq, pol, amap.expert_bytes, link * 1e9, t_layer_fn,
                     depth, base_layer_bytes=base_bytes, hbm_bps=hbm_bps,
                     kv_layer_fn=kv_layer_fn, xfer_lat_s=280e-9,
                     t_compute_fn=t_compute_fn, t0_s=T0_US * 1e-6)

        row = dict(tag=tag, batch=batch, phase=phase, dtype=dtype,
                   gpu=gpu_name, hbm_gen=GPU.hbm_gen,
                   cap_kind=cap_kind, cap_frac=round(frac, 4),
                   hbm_gib=round(hbm_gib, 3), cache_slots=eff_slots,
                   total_slots=total_slots, link_gbps=link,
                   policy=pol_name, depth=depth,
                   hit_rate=r["hit_rate"], total_s=r["total_s"],
                   stall_s=r["stall_s"], compute_s=r["compute_s"],
                   stall_frac=r["stall_fraction"],
                   bytes_fetched=r["bytes_fetched"],
                   bytes_hbm_read=r["bytes_hbm_read"],
                   layers=r["layers_executed"])
        for e_link in E_LINK_CTRL_SWEEP:
            m = EnergyModel(GPU.hbm_gen, stacks * 16, 8, e_link)
            e = m.energy(r["bytes_hbm_read"],
                         r["bytes_fetched"] + r["bytes_hbm_write_kv"],
                         r["bytes_fetched"], r["total_s"],
                         u_hbm=r["u_hbm"], u_ddr5=r["u_ddr5"])
            k = f"e{int(e_link)}"
            row[f"energy_j_{k}"] = e["e_total_j"]
            row[f"edp_js_{k}"] = e["edp_js"]
            row[f"power_w_{k}"] = e["avg_power_w"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT / "sweep.parquet"))
    a = ap.parse_args()

    # Default sweep covers the MoE models only. Qwen2.5-3B-dense is the
    # no-sparsity CONTROL and belongs in its own file (results/dense.parquet)
    # -- mixing it into sweep.parquet silently pollutes any aggregate taken
    # over "the models", which is how it is normally read.  Request it with
    # an explicit --tag.
    tags = [a.tag] if a.tag else [x for x in TAGS if x != "Qwen2.5-3B-dense"]
    jobs = list(itertools.product(tags, BATCHES, PHASES, DTYPES, GPUS_SWEPT))
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    all_rows = []
    with Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_group, jobs)):
            all_rows.extend(rows)
            print(f"[{i+1}/{len(jobs)}] +{len(rows)} rows "
                  f"({len(all_rows)} total, {time.time()-t0:.0f}s)",
                  flush=True)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_parquet(a.out)
    print(f"\n{len(df)} rows -> {a.out}  ({time.time()-t0:.0f}s)")
    # quick sanity surface
    if len(df):
        d = df[(df.phase == "decode") & (df.dtype == "fp16")]
        print("\nsanity: decode fp16, median slowdown vs all-resident by policy")
        base = d[d.cap_frac >= 1.0].groupby("tag").total_s.min()
        for p in POLICIES:
            sub = d[(d.policy == p) & (d.cap_frac == 0.25)
                    & (d.link_gbps == 50.7) & (d.depth == 0)]
            if len(sub):
                sl = (sub.set_index("tag").total_s / base).median()
                print(f"  {p:<10} {sl:6.1f}x @ 25% capacity")


if __name__ == "__main__":
    main()
