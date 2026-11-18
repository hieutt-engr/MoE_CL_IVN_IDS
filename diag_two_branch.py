#!/usr/bin/env python3
"""
diag_two_branch.py: does each branch do what the design says? (inference only)

Loads the last checkpoint of a run and its data, and measures on the test split:

  A. null expert   behavior-output norm of benign vs attacked nodes, and the real-expert
                   weight they receive. A working null expert gives benign nodes a norm
                   close to 0 (ratio benign/attacked << 1); ratio ~1 means LayerNorm
                   restored the expert output and "null" exists only in the routing.
  B. branches off  macro F1 / normal FPR with the identity branch zeroed (in
                   distribution: identity dropout does this in training) and with the
                   behavior branch zeroed (out of distribution: read as an upper bound
                   of its contribution).
  C. ID reliance   the same metrics after permuting the CAN-ID indices and after mapping
                   every ID to the padding index. Works for every architecture, including
                   the standard MoE (std_moe).
  D. probes        linear probes fitted on val nodes, scored on test nodes (balanced
                   accuracy), for the representations: behavior input xb, identity
                   branch, behavior branch before and after normalisation. Tasks:
                   which CAN ID (20 most frequent, benign nodes), attacked or not,
                   attack class of attacked nodes. Design expectation: the identity
                   branch knows the ID; the behavior branch does not (near chance) but
                   separates attacked nodes and classes.

    python -u diag_two_branch.py --run save/seeds_memory/impala/bb_res_10k_s42 --out diag_bb_s42.json
    python -u diag_two_branch.py --run save/ablation_final/impala/std_moe_s42  --out diag_std_s42.json

Run from the project root (it imports train_continual_window.py).
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import train_continual_window as T


# ----------------------------------------------------------------------------- loading
def latest_ckpt(run_dir):
    for ph in ("phase5", "phase4", "phase3", "phase2", "phase1"):
        p = Path(run_dir) / f"{ph}_best.pth"
        if p.exists():
            return p
    raise FileNotFoundError(f"no phase checkpoint in {run_dir}")


def load_model(ck_path, device):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**ck["args"])
    meta = T.load_metadata(Path(a.data_dir) / "metadata.json")
    model = T.build_model(a, meta).to(device)
    if hasattr(model, "moe"):
        while len(model.moe.experts) < ck.get("num_experts", 0):
            model.expand_expert_pool(device=device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, a, meta


def pack(windows, data_dir, split, device, args):
    T._PACKS.append(T.PackedSplit(windows, Path(data_dir) / f"packed_{split}.pt", device,
                                  src_path=Path(data_dir) / f"windows_{split}.pt",
                                  placement=args.fast_data, num_workers=0, log=print))


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def run(model, loader, device, details=False, ids_fn=None):
    is_beh = getattr(model, "needs_behavior", False)
    logits, ys = [], []
    nodes = {k: [] for k in ("identity", "behavior_raw", "behavior", "real_w", "atk", "ny",
                             "ids", "xb", "top1")}
    for batch in loader:
        ids = batch["ids"].to(device)
        if ids_fn is not None:
            ids = ids_fn(ids)
        x, adj, n = batch["x"].to(device), batch["adj"].to(device), batch["n"].to(device)
        y = batch["y"].to(device)
        if is_beh:
            kw = {"xb": batch["xb"].to(device), "node_atk": batch["node_atk"].to(device)}
            if hasattr(model, "window_dim"):
                kw["wb"] = batch["wb"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model.forward_padded(x, ids, adj, n, return_details=details, **kw)
        else:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model.forward_padded(x, ids, adj, n)
        logits.append(out["logits"].float().cpu())
        ys.append(y.cpu())
        if details and is_beh:
            N = x.size(1)
            mask = torch.arange(N, device=device)[None, :] < n[:, None]
            nodes["identity"].append(out["node_identity"].float().cpu())
            nodes["behavior_raw"].append(out["node_behavior_raw"].float().cpu())
            nodes["behavior"].append(out["node_behavior"].float().cpu())
            nodes["real_w"].append(out["node_real_weight"].float().cpu())
            nodes["top1"].append(out["node_probs"].argmax(1).cpu())
            nodes["atk"].append(out["node_atk"].cpu())
            nodes["ny"].append(y[out["node_graph"]].cpu())
            nodes["ids"].append(batch["ids"].to(device)[mask].cpu())
            nodes["xb"].append(batch["xb"].to(device)[mask].float().cpu())
    res = {"logits": torch.cat(logits).numpy(), "y": torch.cat(ys).numpy()}
    if details and is_beh:
        res["nodes"] = {k: torch.cat(v).numpy() for k, v in nodes.items()}
    return res


CLASS_NAMES = ["normal", "dos", "fuzzing", "gear", "interval", "rpm", "speed",
               "standstill", "systematic", "combined"]


def metrics(res, classes):
    y, p = res["y"], res["logits"].argmax(1)
    f1 = {}
    for c in classes:
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f1[CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)] = float(2 * tp / max(2 * tp + fp + fn, 1))
    normal = y == 0
    return {"macro_f1": float(np.mean(list(f1.values()))),
            "normal_fpr": float((p[normal] != 0).mean()) if normal.any() else float("nan"),
            "detection": float((p[~normal] != 0).mean()) if (~normal).any() else float("nan"),
            "per_class_f1": f1}


# ----------------------------------------------------------------------------- probes
def probe(Xtr, ytr, Xte, yte, device, epochs=300):
    """Multinomial logistic regression (standardised inputs), balanced accuracy on test."""
    labels = np.unique(np.concatenate([ytr, yte]))
    lut = {v: i for i, v in enumerate(labels)}
    ytr_i = torch.tensor([lut[v] for v in ytr], device=device)
    yte_i = np.array([lut[v] for v in yte])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    Xte_t = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=device)
    W = torch.nn.Linear(Xtr.shape[1], len(labels)).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=1e-2, weight_decay=1e-4)
    counts = torch.bincount(ytr_i, minlength=len(labels)).float()
    weight = (counts.sum() / (len(labels) * counts.clamp(min=1)))
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(W(Xtr_t), ytr_i, weight=weight).backward()
        opt.step()
    pred = W(Xte_t).argmax(1).cpu().numpy()
    acc = [np.mean(pred[yte_i == k] == k) for k in range(len(labels)) if (yte_i == k).any()]
    return float(np.mean(acc)), 1.0 / len(labels)


def subsample(idx, n, rng):
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)


def probes(val_nodes, test_nodes, device, n_max, seed=0):
    rng = np.random.default_rng(seed)
    reps = ("xb", "identity", "behavior_raw", "behavior")
    out = {}

    def task(name, sel_fn, label_fn):
        itr = subsample(np.where(sel_fn(val_nodes))[0], n_max, rng)
        ite = subsample(np.where(sel_fn(test_nodes))[0], n_max, rng)
        if len(itr) < 50 or len(ite) < 50:
            return
        ytr, yte = label_fn(val_nodes)[itr], label_fn(test_nodes)[ite]
        out[name] = {}
        for r in reps:
            acc, chance = probe(val_nodes[r][itr], ytr, test_nodes[r][ite], yte, device)
            out[name][r] = round(acc, 4)
        out[name]["chance"] = round(chance, 4)

    benign_v = val_nodes["atk"] <= 0
    ids_b, cnt = np.unique(val_nodes["ids"][benign_v], return_counts=True)
    top = ids_b[np.argsort(-cnt)[:20]]
    task("can_id_top20 (benign nodes)", lambda d: (d["atk"] <= 0) & np.isin(d["ids"], top),
         lambda d: d["ids"])
    task("attacked_or_not", lambda d: np.ones(len(d["atk"]), bool),
         lambda d: (d["atk"] > 0).astype(int))
    task("attack_class (attacked nodes)", lambda d: d["atk"] > 0, lambda d: d["ny"])
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory (uses its last phase checkpoint)")
    ap.add_argument("--ckpt", help="explicit checkpoint path")
    ap.add_argument("--data_dir", default=None, help="default: the data_dir stored in the checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--probe_nodes", type=int, default=40000)
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    ck_path = Path(a.ckpt) if a.ckpt else latest_ckpt(a.run)
    model, margs, meta = load_model(ck_path, device)
    data_dir = Path(a.data_dir or margs.data_dir)
    margs.batch_size, margs.num_workers = a.batch_size, 0
    margs.fast_data = "gpu" if device.type == "cuda" else "cpu"
    is_beh = getattr(model, "needs_behavior", False)
    print(f"{ck_path} | arch {margs.arch} | experts "
          f"{len(model.moe.experts) if hasattr(model, 'moe') else '-'} | null_scale "
          f"{getattr(model, 'null_scale', '-')} | mask_unknown_ids {getattr(model, 'mask_unknown_ids', '-')}")

    test = T.load_windows(data_dir / "windows_test.pt")
    val = T.load_windows(data_dir / "windows_val.pt") if is_beh else None
    T._PACKS.clear()
    pack(test, data_dir, "test", device, margs)
    if val is not None:
        pack(val, data_dir, "val", device, margs)
    classes = sorted({w["label"] for w in test})
    report = {"checkpoint": str(ck_path), "arch": margs.arch}

    base = run(model, T.make_loader(test, margs), device, details=is_beh)
    report["baseline"] = metrics(base, classes)
    print("baseline:", {k: v for k, v in report["baseline"].items() if k != "per_class_f1"})

    # A. null expert
    if is_beh:
        nd = base["nodes"]
        norm = np.linalg.norm(nd["behavior"], axis=1)
        ben, att = nd["atk"] <= 0, nd["atk"] > 0
        nul = nd["top1"] == 0
        report["A_null"] = {
            "real_weight_benign": float(nd["real_w"][ben].mean()),
            "real_weight_attacked": float(nd["real_w"][att].mean()),
            "behavior_norm_benign": float(norm[ben].mean()),
            "behavior_norm_attacked": float(norm[att].mean()),
            "behavior_norm_null_routed": float(norm[nul].mean()) if nul.any() else None,
            "ratio_benign_over_attacked": float(norm[ben].mean() / max(norm[att].mean(), 1e-9)),
        }
        print("A. null expert:", report["A_null"])

    # B. branches off
    if is_beh:
        report["B_branch_off"] = {}
        for br in ("identity", "behavior"):
            model.branch_off = br
            report["B_branch_off"][br] = metrics(run(model, T.make_loader(test, margs), device), classes)
            print(f"B. {br} branch off:", {k: v for k, v in report["B_branch_off"][br].items() if k != "per_class_f1"})
        model.branch_off = None

    # C. reliance on CAN-ID identity
    V = meta["vocab_size"]
    g = torch.Generator().manual_seed(0)
    perm = torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(V - 1, generator=g)]).to(device)
    report["C_id_reliance"] = {
        "permuted_ids": metrics(run(model, T.make_loader(test, margs), device,
                                    ids_fn=lambda i: perm[i.long().clamp(0, V - 1)]), classes),
        "all_ids_padding": metrics(run(model, T.make_loader(test, margs), device,
                                       ids_fn=lambda i: torch.zeros_like(i)), classes),
    }
    print("C. ID reliance:", report["C_id_reliance"])

    # D. probes
    if is_beh:
        vres = run(model, T.make_loader(val, margs), device, details=True)
        report["D_probes"] = probes(vres["nodes"], base["nodes"], device, a.probe_nodes)
        for task, r in report["D_probes"].items():
            print(f"D. {task}: " + ", ".join(f"{k} {v}" for k, v in r.items()))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
