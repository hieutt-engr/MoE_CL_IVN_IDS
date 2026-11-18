#!/usr/bin/env python3
"""
summarize_baselines.py: continual-learning baselines vs the final model A (and the
one-branch MoE if available), mean +- std over seeds, phase order variant.

    python summarize_baselines.py [--smoke]
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SAVE = Path("save")
root = SAVE / ("baselines_smoke" if "--smoke" in sys.argv else "baselines") / "impala"
ORDER = "variant"
NAMES = {"finetune": "Fine-tune (lower bound)", "er": "ER", "cbrs": "ER + class-balanced memory (CBRS)",
         "ewc": "EWC + replay", "lwf": "LwF + replay", "icarl": "iCaRL", "derpp": "DER++",
         "der": "DER (expandable)", "joint": "Joint (upper bound)"}


def load(d):
    d = Path(d)
    if not (d / "cl_summary.json").exists():
        return None
    cl = json.load(open(d / "cl_summary.json"))
    res = json.load(open(d / "results.json"))
    old = [c for c in res[0]["per_attack"]]                 # phase-1 classes
    return {"macro F1": cl["final_macro_f1"], "normal FPR": cl["final_normal_fpr"],
            "forgetting mean": cl["forgetting_mean"], "forgetting max": cl["forgetting_max"],
            "new-class F1": cl["new_class_f1_when_learned"],
            "phase-1 classes F1 (final)": float(np.mean([res[-1]["per_attack"][c] for c in old])),
            "params / experts": res[-1].get("params") or res[-1].get("num_experts")}


rows = []
for d in sorted(glob.glob(str(root / f"*_{ORDER}_s*"))):
    m = re.match(rf"(\w+?)_{ORDER}_s(\d+)(_la)?$", Path(d).name)
    r = load(d)
    if m and r:
        rows.append({"method": NAMES.get(m.group(1), m.group(1)) + (" (+LA)" if m.group(3) else ""),
                     "seed": int(m.group(2)), **r})
for d in sorted(glob.glob(str(SAVE / "v5/impala" / f"A_{ORDER}_s*"))):
    r = load(d)
    if r:
        rows.append({"method": "Ours: two-branch MoE (A)", "seed": int(d.rsplit("_s", 1)[1]), **r})
for d in sorted(glob.glob(str(SAVE / "v3/impala/std_moe_s*"))):
    r = load(d)
    if r:
        rows.append({"method": "One-branch MoE (ID-routed)", "seed": int(d.rsplit("_s", 1)[1]), **r})

df = pd.DataFrame(rows)
if df.empty:
    sys.exit(f"No finished runs under {root}")
cols = ["macro F1", "normal FPR", "forgetting mean", "forgetting max", "new-class F1", "phase-1 classes F1 (final)"]
pd.set_option("display.width", 220)
print("Per run:")
print(df.sort_values(["method", "seed"]).round(4).to_string(index=False))
g = df.groupby("method")
table = pd.DataFrame({c: g[c].mean().map("{:.4f}".format) + " ± " + g[c].std().fillna(0).map("{:.4f}".format)
                      for c in cols})
table["seeds"] = g.size()
table["params / experts (last)"] = g["params / experts"].last()
table = table.loc[g["macro F1"].mean().sort_values(ascending=False).index]
print("\nMean ± std over seeds:")
print(table.to_string())
out = root / "summary_baselines.csv"
table.to_csv(out); df.to_csv(root / "summary_baselines_runs.csv", index=False)
print(f"\nSaved {out}")
