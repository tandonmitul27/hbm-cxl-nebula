"""Integrity checks on the routing logs shipped in data/routing/.

Phase 1 only claims one thing: for every (token, layer) it recorded the set of
experts the router actually selected.  This verifies that claim structurally --
independently of the row-count arithmetic in manage_model.py, which only checks
totals and would miss misalignment, duplication, or gaps.

    python check_integrity.py            # all tags
    python check_integrity.py --tag OLMoE-1B-7B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE
GEO  = ROOT / "mapping" / "geometry"
PREFILL, DECODE = 0, 1


def check_run(path: Path, meta: dict, run: dict) -> list[str]:
    g = meta["geometry"]
    n_exp, k = g["num_experts"], g["top_k"]
    moe_layers = set(g["moe_layers"])
    P, G, B = run["prompt_len"], run["gen_len"], run["batch"]
    bad = []

    t = pq.read_table(path)
    d = {c: t.column(c).to_numpy() for c in t.column_names}
    phase, step, layer = d["phase"], d["step"], d["layer"]
    seq, pos, exp, w = d["seq_id"], d["pos"], d["expert"], d["weight"]

    # --- expert ids in range -------------------------------------------
    if exp.min() < 0 or exp.max() >= n_exp:
        bad.append(f"expert id out of range [{exp.min()},{exp.max()}] vs N={n_exp}")

    # --- exactly the MoE layers, no others ------------------------------
    seen_layers = set(np.unique(layer).tolist())
    if seen_layers != moe_layers:
        bad.append(f"layers recorded {sorted(seen_layers)[:5]}... != "
                   f"MoE layers {sorted(moe_layers)[:5]}...")

    # --- exactly k rows per (seq, layer, pos), all distinct --------------
    # The group key must include pos: during prefill every token shares
    # step=0, so keying on step alone merges the whole prompt into one group.
    # pos is unique per token within a sequence and prefill/decode ranges are
    # disjoint (decode starts at prompt_len), so (seq, layer, pos) is exact.
    order = np.lexsort((exp, pos, layer, seq))
    s_, l_, p_, e_ = seq[order], layer[order], pos[order], exp[order]
    new_grp = np.r_[True, (s_[1:] != s_[:-1]) | (l_[1:] != l_[:-1])
                    | (p_[1:] != p_[:-1])]
    starts = np.flatnonzero(new_grp)
    counts = np.diff(np.r_[starts, s_.size])
    if not np.all(counts == k):
        wrong = int((counts != k).sum())
        bad.append(f"{wrong} groups do not have exactly k={k} rows "
                   f"(sizes seen {np.unique(counts)[:5]})")
    # same group (no boundary) AND same expert => the same expert twice
    dup = (~new_grp[1:]) & (e_[1:] == e_[:-1])
    if dup.any():
        bad.append(f"{int(dup.sum())} duplicate experts within a top-k set")

    # --- prefill covers every position exactly once ----------------------
    pm = phase == PREFILL
    if pm.any():
        want = B * len(moe_layers) * k
        u, c = np.unique(pos[pm], return_counts=True)
        if u.min() != 0 or u.max() != P - 1 or u.size != P:
            bad.append(f"prefill positions cover {u.min()}..{u.max()} "
                       f"({u.size} distinct), expected 0..{P-1}")
        elif not np.all(c == want):
            bad.append(f"prefill position counts vary {c.min()}..{c.max()}, "
                       f"expected {want} (chunking dropped or duplicated tokens)")

    # --- decode position must be prompt_len + step -----------------------
    dm = phase == DECODE
    if dm.any():
        off = pos[dm] - step[dm]
        if not np.all(off == P):
            bad.append(f"decode pos-step offset {np.unique(off)[:4]} != prompt_len {P}")
        u = np.unique(step[dm])
        if u.size != G or u.min() != 0 or u.max() != G - 1:
            bad.append(f"decode steps {u.min()}..{u.max()} ({u.size}) != 0..{G-1}")

    # --- batch coverage ---------------------------------------------------
    if np.unique(seq).size != B:
        bad.append(f"{np.unique(seq).size} sequences present, expected {B}")

    # --- weights sane ------------------------------------------------------
    if not np.isfinite(w).all():
        bad.append("non-finite router weights")
    elif w.min() < 0:
        bad.append(f"negative router weight {w.min()}")

    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    # underscore-prefixed files are auxiliary sidecars (e.g. _attention_shapes),
    # not per-model extraction records
    index = json.loads((DATA / "routing" / "index.json").read_text())
    tags = [a.tag] if a.tag else sorted(index)
    total_bad = 0
    for tag in tags:
        # geometry is single-sourced in mapping/geometry/; the run list is
        # the index that ships beside the logs it describes.
        meta = json.loads((GEO / f"{tag}.json").read_text())
        meta["runs"] = index[tag]
        g = meta["geometry"]
        print(f"\n=== {tag}  (N={g['num_experts']} k={g['top_k']} "
              f"moe_layers={len(g['moe_layers'])}/{g['num_layers']}"
              f"{', shared=' + str(g['num_shared_experts']) if g['num_shared_experts'] else ''})")
        for run in meta["runs"]:
            if not run.get("ok"):
                continue
            p = DATA / "routing" / tag / run["file"]
            probs = check_run(p, meta, run)
            total_bad += len(probs)
            status = "OK " if not probs else "BAD"
            print(f"  [{status}] {run['file']:<28} rows={run['rows']:>10,}")
            for x in probs:
                print(f"          - {x}")
    print(f"\n{'ALL CHECKS PASSED' if total_bad == 0 else f'{total_bad} PROBLEM(S)'}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
