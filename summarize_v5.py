#!/usr/bin/env python3
"""
summarize_v5.py: tables for the final configuration D vs A (same without the balance loss).

    python summarize_v5.py            # prints the tables, writes save/v5/summary_*.csv

Seed 42 comes from save/v4/impala, seeds 43 and 44 from save/v5/impala.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SAVE = Path("save")
SEED42 = {"A": SAVE / "v4/impala/A_rulemax_e4_s42", "D": SAVE / "v4/impala/D_rulemax_e4_bal_s42"}
CLASSES = ["normal", "dos", "fuzzing", "gear", "interval", "rpm", "speed", "standstill", "systematic", "combined"]
PHASE1_ATTACKS = {"variant": ["dos", "fuzzing", "gear", "interval", "rpm"],
                  "late_injection": ["dos", "gear", "interval", "rpm"],
                  "late_spoof": ["dos", "fuzzing", "interval", "systematic"]}


def load(d):
    d = Path(d)
    if not (d / "results.json").exists():
        return None
    res = json.load(open(d / "results.json"))
    cl = json.load(open(d / "cl_summary.json")) if (d / "cl_summary.json").exists() else {}
    return res, cl


def phase1_purity(run_dir, attacks):
    """Mean over phase-1 attack classes of the routing mass of their main real expert."""
    p = Path(run_dir) / "preds_phase1.npz"
    if not p.exists():
        return np.nan, ""
    z = np.load(p)
    atk, ny, real = z["node_atk"] > 0, z["node_ny"], z["node_probs"][:, 1:].astype(np.float32)
    conc, main = [], {}
    for c in attacks:
        sel = atk & (ny == CLASSES.index(c))
        if sel.sum() >= 20:
            m = real[sel].mean(0); conc.append(m.max()); main[c] = int(m.argmax()) + 1
    groups = {}
    for c, e in main.items():
        groups.setdefault(e, []).append(c)
    shared = "; ".join(f"E{e}: {'+'.join(cs)}" for e, cs in groups.items() if len(cs) > 1)
    return float(np.mean(conc)) if conc else np.nan, shared or "none"


def row(name, run_dir, order="variant"):
    r = load(run_dir)
    if r is None:
        return None
    res, cl = r
    last = res[-1]
    sp1 = (res[0].get("specialization") or {}); sp5 = (last.get("specialization") or {})
    conc, shared = phase1_purity(run_dir, PHASE1_ATTACKS[order])
    diag = None
    for dd in (SAVE / "v5/diag", SAVE / "v4/diag"):
        f = dd / f"{Path(run_dir).name}.json"
        if f.exists():
            diag = json.load(open(f)); break
    return {"run": name, "order": order, "macro F1": last["overall"]["macro_f1"],
            "normal FPR": last["overall"]["normal_fpr"], "forgetting mean": cl.get("forgetting_mean"),
            "experts": last.get("num_experts"),
            "NMI P1": sp1.get("nmi_active_class"), "NMI P5": sp5.get("nmi_active_class"),
            "P1 concentration": conc, "P1 shared experts": shared,
            "F1 identity off": (diag or {}).get("B_branch_off", {}).get("identity", {}).get("macro_f1"),
            "decisions": " / ".join((x.get("decision_info") or {}).get("decision", "")[:6] for x in res[1:]),
            **{f"F1 {c}": last["per_attack"].get(c) for c in CLASSES[1:]}}


rows = []
for cfg in ("A", "D"):
    rows.append(row(f"{cfg}_s42", SEED42[cfg]))
    for s in (43, 44):
        rows.append(row(f"{cfg}_s{s}", SAVE / f"v5/impala/{cfg}_variant_s{s}"))
seeds = pd.DataFrame([r for r in rows if r]).set_index("run")
seeds["config"] = [i[0] for i in seeds.index]

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
main_cols = ["macro F1", "normal FPR", "forgetting mean", "experts", "NMI P1", "NMI P5",
             "P1 concentration", "F1 identity off"]
print("\n=== Phase order variant, per seed ===")
print(seeds[main_cols + ["P1 shared experts", "decisions"]].round(4).to_string())

agg = seeds.groupby("config")[main_cols].agg(["mean", "std", "count"])
print("\n=== Mean +- std over seeds (A: no balance loss, D: final) ===")
for cfg, g in seeds.groupby("config"):
    parts = [f"{c} {g[c].mean():.4f} +- {g[c].std(ddof=1) if len(g) > 1 else float('nan'):.4f}" for c in main_cols]
    print(f"{cfg} (n={len(g)}): " + " | ".join(parts))
if {"A", "D"} <= set(seeds["config"]):
    d = seeds[seeds.config == "D"]; a = seeds[seeds.config == "A"]
    for c in ("macro F1", "normal FPR", "NMI P1"):
        diff = d[c].mean() - a[c].mean()
        sd = np.sqrt((d[c].var(ddof=1) + a[c].var(ddof=1)) / 2) if min(len(d), len(a)) > 1 else np.nan
        print(f"  D - A on {c}: {diff:+.4f}  (pooled std {sd:.4f}; |diff| > 2 std suggests a real difference)")

rows = [row("D_variant_s42", SEED42["D"])]
for order in ("late_injection", "late_spoof"):
    for cfg in ("D", "A"):
        rows.append(row(f"{cfg}_{order}_s42", SAVE / f"v5/impala/{cfg}_{order}_s42", order))
    rows.append(row(f"old_rule_{order}_s42", SAVE / f"v3/impala/final_{order}_s42", order))
orders = pd.DataFrame([r for r in rows if r]).set_index("run")
print("\n=== Phase orders (seed 42) ===")
print(orders[["order"] + main_cols + ["P1 shared experts", "decisions"]].round(4).to_string())

out = SAVE / "v5"; out.mkdir(parents=True, exist_ok=True)
seeds.to_csv(out / "summary_seeds.csv"); orders.to_csv(out / "summary_orders.csv")
print(f"\nSaved {out / 'summary_seeds.csv'} and {out / 'summary_orders.csv'}")
