#!/usr/bin/env python3
"""
check_vehicles.py: data inventory and cross-vehicle CAN ID overlap (no training).

Reads the raw attack CSVs of every vehicle under --data_root (either directly in
data/<vehicle>/ or in data/<vehicle>/merged/) and answers four questions:

  1. inventory   which attack files each vehicle has, messages, attack frames,
                 duration, message rate
  2. ID overlap  how many CAN IDs seen in normal traffic are shared by each pair
                 of vehicles, and how much of the traffic they carry
  3. periods     for shared IDs, whether the sending period is the same
  4. targets     which CAN IDs each attack writes to on each vehicle, whether
                 they are IDs of normal traffic, and whether the main target is
                 the same ID across vehicles

    python check_vehicles.py                          # all vehicles in ./data
    python check_vehicles.py --max_rows 200000        # quick test on the first rows

Outputs (CSV + JSON) go to --out (default ./data_check).
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ATTACKS = ["dos", "fuzzing", "gear", "interval", "rpm", "speed", "standstill",
           "systematic", "combined"]


# ----------------------------------------------------------------------------- helpers
def find_csvs(vdir):
    sub = vdir / "merged"
    base = sub if sub.is_dir() and any(sub.glob("*.csv")) else vdir
    out = {}
    for f in sorted(base.glob("*.csv")):
        stem = f.stem.lower()
        if stem in ATTACKS:
            out[stem] = f
    return out


def resolve_columns(path):
    cols = list(pd.read_csv(path, nrows=0).columns)
    low = {c.lower().strip(): c for c in cols}

    def pick(*names):
        for n in names:
            for k, c in low.items():
                if k == n:
                    return c
        for n in names:
            for k, c in low.items():
                if n in k:
                    return c
        return None

    m = {"ts": pick("timestamp", "time"), "id": pick("arbitration_id", "can_id", "id"),
         "data": pick("data_field", "data", "payload"), "atk": pick("attack", "label", "flag")}
    missing = [k for k, v in m.items() if v is None and k != "data"]
    if missing:
        raise ValueError(f"{path.name}: cannot find columns {missing} in {cols}")
    return m


def parse_ids(series):
    """Hex or int CAN IDs -> int, mapping each distinct value once."""
    uniq = pd.unique(series.astype(str))
    lut = {}
    for u in uniq:
        s = u.strip().lower().replace("0x", "")
        try:
            lut[u] = int(s, 16)
        except ValueError:
            lut[u] = -1
    return series.astype(str).map(lut).to_numpy()


# ----------------------------------------------------------------------------- per file
def scan_file(path, cols, chunksize, max_rows):
    n = n_atk = 0
    t_min, t_max = np.inf, -np.inf
    norm_cnt, atk_cnt = {}, {}
    period_samples = {}          # id -> list of per-chunk median gaps (normal frames)
    read = 0
    usecols = [cols["ts"], cols["id"], cols["atk"]]
    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        if max_rows and read >= max_rows:
            break
        if max_rows:
            ch = ch.iloc[: max_rows - read]
        read += len(ch)
        ts = pd.to_numeric(ch[cols["ts"]], errors="coerce").to_numpy()
        ids = parse_ids(ch[cols["id"]])
        atk = pd.to_numeric(ch[cols["atk"]], errors="coerce").fillna(0).to_numpy() > 0
        n += len(ch)
        n_atk += int(atk.sum())
        if len(ts):
            t_min, t_max = min(t_min, np.nanmin(ts)), max(t_max, np.nanmax(ts))
        for mask, store in ((~atk, norm_cnt), (atk, atk_cnt)):
            u, c = np.unique(ids[mask], return_counts=True)
            for i, k in zip(u.tolist(), c.tolist()):
                store[i] = store.get(i, 0) + k
        nd = pd.DataFrame({"id": ids[~atk], "ts": ts[~atk]})
        if len(nd):
            gaps = nd.groupby("id")["ts"].diff()
            med = gaps.groupby(nd["id"]).median().dropna()
            for i, g in med.items():
                if g > 0:
                    period_samples.setdefault(int(i), []).append(float(g))
    dur = float(t_max - t_min) if np.isfinite(t_min) else 0.0
    return {"messages": n, "attack_frames": n_atk, "duration_s": dur,
            "norm_cnt": norm_cnt, "atk_cnt": atk_cnt, "period_samples": period_samples}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--vehicles", nargs="*", default=None, help="default: every folder in data_root")
    ap.add_argument("--out", default="./data_check")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--max_rows", type=int, default=0, help="per file, 0 = all rows")
    ap.add_argument("--top_targets", type=int, default=3)
    a = ap.parse_args()

    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vdirs = [root / v for v in a.vehicles] if a.vehicles else sorted(
        p for p in root.iterdir() if p.is_dir() and find_csvs(p))
    if not vdirs:
        raise SystemExit(f"no vehicle folders with attack CSVs under {root}")

    inv_rows, V = [], {}
    for vdir in vdirs:
        name = vdir.name
        files = find_csvs(vdir)
        missing = [x for x in ATTACKS if x not in files]
        print(f"\n=== {name}  ({len(files)} attack files{', missing ' + str(missing) if missing else ''})")
        veh = {"norm_cnt": {}, "duration": 0.0, "period_samples": {}, "attacks": {}}
        for atk, f in files.items():
            cols = resolve_columns(f)
            r = scan_file(f, cols, a.chunksize, a.max_rows)
            for i, c in r["norm_cnt"].items():
                veh["norm_cnt"][i] = veh["norm_cnt"].get(i, 0) + c
            for i, s in r["period_samples"].items():
                veh["period_samples"].setdefault(i, []).extend(s)
            veh["duration"] += r["duration_s"]
            veh["attacks"][atk] = r["atk_cnt"]
            rate = r["messages"] / r["duration_s"] if r["duration_s"] > 0 else float("nan")
            inv_rows.append({"vehicle": name, "attack": atk, "file": f.name,
                             "messages": r["messages"], "attack_frames": r["attack_frames"],
                             "attack_ratio": r["attack_frames"] / max(r["messages"], 1),
                             "duration_s": round(r["duration_s"], 1), "msg_per_s": round(rate, 1),
                             "ids_normal": len(r["norm_cnt"]), "ids_attack": len(r["atk_cnt"])})
            print(f"  {atk:<11} {r['messages']:>10,} msgs  {r['attack_frames']:>8,} attack "
                  f"({inv_rows[-1]['attack_ratio']:.2%})  {r['duration_s']:>8.0f} s  "
                  f"{len(r['norm_cnt'])} normal IDs")
        veh["ids"] = set(veh["norm_cnt"])
        veh["period"] = {i: float(np.median(s)) for i, s in veh["period_samples"].items()}
        V[name] = veh
        print(f"  -> {len(veh['ids'])} CAN IDs in normal traffic")
    pd.DataFrame(inv_rows).to_csv(out / "inventory.csv", index=False)

    # ---- 2 + 3. pairwise ID overlap and period similarity
    ov_rows = []
    print("\n=== CAN ID overlap (normal traffic)")
    print(f"  {'pair':<52}{'shared':>7}{'Jaccard':>9}{'traffic A':>11}{'traffic B':>11}"
          f"{'period ±10%':>13}{'median |log ratio|':>20}")
    for A, B in combinations(V, 2):
        ia, ib = V[A]["ids"], V[B]["ids"]
        sh = ia & ib
        cov = lambda v, s: sum(V[v]["norm_cnt"][i] for i in s) / max(sum(V[v]["norm_cnt"].values()), 1)
        pa = np.array([V[A]["period"].get(i, np.nan) for i in sh])
        pb = np.array([V[B]["period"].get(i, np.nan) for i in sh])
        ok = np.isfinite(pa) & np.isfinite(pb) & (pa > 0) & (pb > 0)
        lr = np.abs(np.log(pa[ok] / pb[ok])) if ok.any() else np.array([])
        row = {"vehicle_a": A, "vehicle_b": B, "ids_a": len(ia), "ids_b": len(ib), "shared": len(sh),
               "jaccard": len(sh) / max(len(ia | ib), 1),
               "traffic_share_a": cov(A, sh), "traffic_share_b": cov(B, sh),
               "shared_with_period": int(ok.sum()),
               "period_within_10pct": float((lr < np.log(1.1)).mean()) if len(lr) else float("nan"),
               "median_abs_log_period_ratio": float(np.median(lr)) if len(lr) else float("nan")}
        ov_rows.append(row)
        print(f"  {A + ' / ' + B:<52}{row['shared']:>7}{row['jaccard']:>9.2f}"
              f"{row['traffic_share_a']:>11.1%}{row['traffic_share_b']:>11.1%}"
              f"{row['period_within_10pct']:>13.1%}{row['median_abs_log_period_ratio']:>20.3f}")
    pd.DataFrame(ov_rows).to_csv(out / "id_overlap.csv", index=False)

    per_rows = []
    all_ids = sorted(set().union(*(V[v]["ids"] for v in V)))
    for i in all_ids:
        r = {"can_id": f"{i:03X}"}
        for v in V:
            r[f"period_ms[{v}]"] = round(V[v]["period"][i] * 1000, 2) if i in V[v]["period"] else None
        per_rows.append(r)
    pd.DataFrame(per_rows).to_csv(out / "periods_by_id.csv", index=False)

    # ---- 4. attack targets
    tg_rows = []
    print(f"\n=== Attack targets (top {a.top_targets} CAN IDs by attack frames; "
          f"'known' = share of attack frames on IDs of the vehicle's normal traffic)")
    for atk in ATTACKS:
        print(f"  {atk}")
        tops = {}
        for v in V:
            cnt = V[v]["attacks"].get(atk)
            if not cnt:
                print(f"    {v:<30} (no file)")
                continue
            tot = sum(cnt.values())
            known = sum(c for i, c in cnt.items() if i in V[v]["ids"]) / max(tot, 1)
            top = sorted(cnt.items(), key=lambda kv: -kv[1])[: a.top_targets]
            tops[v] = top[0][0] if top else None
            desc = ", ".join(f"{i:03X} ({c / tot:.0%})" for i, c in top)
            print(f"    {v:<30} {len(cnt):>5} IDs  known {known:>5.1%}  top: {desc}")
            tg_rows.append({"attack": atk, "vehicle": v, "attack_frames": tot, "n_target_ids": len(cnt),
                            "share_on_known_ids": known,
                            **{f"top{k + 1}": f"{i:03X}" for k, (i, _) in enumerate(top)},
                            **{f"top{k + 1}_share": c / tot for k, (_, c) in enumerate(top)}})
        if len(tops) > 1:
            same = len({t for t in tops.values() if t is not None}) == 1
            print(f"    -> main target {'SAME' if same else 'DIFFERENT'} across vehicles")
    pd.DataFrame(tg_rows).to_csv(out / "attack_targets.csv", index=False)

    json.dump({"vehicles": {v: {"normal_ids": len(V[v]["ids"]), "duration_s": V[v]["duration"]} for v in V},
               "overlap": ov_rows}, open(out / "report.json", "w"), indent=2)
    print(f"\nSaved inventory.csv, id_overlap.csv, periods_by_id.csv, attack_targets.csv, report.json in {out}")


if __name__ == "__main__":
    main()
