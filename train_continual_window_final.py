#!/usr/bin/env python3
"""
train_continual_final.py: entropy-driven expert expansion for IVN IDS.

Consolidates train_continual_clean.py with:
  * fast GPU data path (fast_data.py) and vectorised forward_padded for every
    model (cosine, v2, dense); CPU threads capped, no per-batch host syncs
  * --gpu to pick one card on a multi-GPU server
  * experiment flags for the next round:
      replay       --replay {ratio,class_balanced,buffer,all}, --replay_ratio,
                   --buffer_per_class
      capacity     --train_scope {default,all,backbone}, --freeze_old_gate,
                   --no_balance_on_expand
      bias         --logit_adjust TAU (logit-adjusted CE after expansion)
      schedule     --phase_epochs, --phase_lr_mult, --phase_scheduler,
                   --retrain_epochs (full_retrain upper bound)
  * per-phase test predictions saved to preds_<phase>.npz for analysis

With default flags the training procedure is the one of cosine_s42 (cosine
head, feature distillation 2.0 on all samples, replay = 2 x new data, mean
entropy decision at 10 x baseline). Shuffling comes from torch.randperm, so
runs are reproducible among themselves but not digit-identical to runs made
with the old DataLoader.

Usage:
  python train_continual_final.py --gpu 1 --save_dir ./save/final/impala/base_s42 \\
      --resume_phase1 ./save/clean/impala/cosine_s42/phase1_best.pth
"""

"""
train_continual_window.py: train_continual_dual.py + window-level behavior.

--arch window uses networks/moe_ivn_ids_window.py (router and experts also see
the window behavior vector wb from preprocess_window.py). Each phase also logs
the confusions of the hard class pairs (fuzzing/systematic, rpm/standstill,
combined/spoofing, normal false alarms) on the test set.

train_continual_dual.py: dual-entropy monitoring with ignore / reuse / expand.

Reuse is verified: after training the head on reused experts, the val F1 of the
old classes and of the new class is checked; if an old class drops by more than
--reuse_tol or the new class stays below --reuse_min_f1, the reuse is rolled
back and the phase is learned with a residual expert instead.

Built on train_continual_behavior.py (behavior MoE v2). Two additions:

1. Dual-entropy decision (--decision dual)
   class novelty     cls entropy of the new data vs phase-1 baseline (as before)
   behavior novelty  routing uncertainty measured ONLY on nodes the router
                     treats as abnormal (p_null < 0.5): normalised entropy of the
                     distribution over real experts, vs the phase-1 baseline.
                     Benign nodes, which go confidently to null, no longer dilute it.
   action
     ignore   class not novel                       -> no training (e.g. dos part 2)
     reuse    class novel, behavior known           -> no new expert; train the
              classifier head on reused expert outputs (e.g. standstill, systematic)
     expand   class novel and behavior novel        -> add a residual expert

2. Residual expansion
   --attract_weight / --attract_share: attacked nodes of the new classes must
   give the new expert at least a share of their routing mass (hinge). With
   top_k = 2 a node keeps its old expert for the shared behavior and uses the
   new one for what distinguishes the new attack; old experts stay frozen.
   --old_lr_mult: optional middle ground, old experts trainable at lr * mult.

--decision cls reproduces the previous behaviour (expand whenever the class is new).
"""

import os
import sys


def _early_cpu_threads():
    # must run before numpy/torch import to cap BLAS/OpenMP thread pools
    n = "4"
    if "--cpu_threads" in sys.argv:
        i = sys.argv.index("--cpu_threads")
        if i + 1 < len(sys.argv):
            n = sys.argv[i + 1]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, n)


_early_cpu_threads()

import argparse, atexit, copy, json, shutil, time   # noqa: E402
from pathlib import Path                             # noqa: E402

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn as nn                                # noqa: E402
import torch.nn.functional as F                      # noqa: E402
from torch.utils.data import DataLoader              # noqa: E402
from tqdm import tqdm as _tqdm                       # noqa: E402

from fast_data_window import GPUBatches, PackedSplit, lists_from_padded   # noqa: E402
from networks.moe_ivn_ids_cosine import MoEIVNIDS as MoEIVNIDSCosine  # noqa: E402
from networks.moe_ivn_ids import MoEIVNIDSv2      # noqa: E402
from networks.moe_ivn_ids_behavior import BehaviorMoEIDS   # noqa: E402
from networks.moe_ivn_ids_window import WindowBehaviorMoEIDS   # noqa: E402
from utils import (WindowDataset, collate_fn, compute_metrics, get_class_names,  # noqa: E402
                   load_metadata, load_windows, save_ckpt, set_seed, setup_logger)


# ============================================================
# Phase definitions
# ============================================================

DEFAULT_PHASES = [
    {"name": "phase1", "attacks": [0, 1, 2, 3, 4, 5], "type": "base",
     "desc": "Base: normal + dos + fuzzing + gear + interval + rpm"},
    {"name": "phase2", "attacks": [6], "type": "new", "desc": "New: speed"},
    {"name": "phase3", "attacks": [7], "type": "new", "desc": "New: standstill"},
    {"name": "phase4", "attacks": [8, 9], "type": "new", "desc": "New: systematic + combined"},
]

VARIANT_PHASES = [
    {"name": "phase1", "attacks": [0, 1, 2, 3, 4, 5], "type": "base", "use_dos_part": 1,
     "desc": "Base: normal + dos(part1) + fuzzing + gear + interval + rpm"},
    {"name": "phase2", "attacks": [6], "type": "new", "desc": "New: speed"},
    {"name": "phase3", "attacks": [1], "type": "variant", "use_dos_part": 2,
     "desc": "Variant: dos(part2), same behavior, unseen data"},
    {"name": "phase4", "attacks": [7], "type": "new", "desc": "New: standstill"},
    {"name": "phase5", "attacks": [8, 9], "type": "new", "desc": "New: systematic + combined"},
]

CLASS_NAMES = get_class_names()
NC = len(CLASS_NAMES)


# ============================================================
# Small helpers
# ============================================================

def build_model(args, meta):
    common = dict(vocab_size=meta["vocab_size"], num_features=meta["num_features"],
                  embed_dim=args.embed_dim, num_experts=args.num_experts,
                  top_k=args.top_k, moe_out_dim=args.moe_out_dim, dropout=args.dropout)
    if args.arch == "cosine":
        return MoEIVNIDSCosine(**common)
    if args.arch == "window":
        if "window_dim" not in meta:
            sys.exit("--arch window needs data from preprocess_window.py (no window_dim in metadata)")
        return WindowBehaviorMoEIDS(**common, behavior_dim=meta.get("behavior_dim", 13),
                                    window_dim=meta["window_dim"],
                                    null_weight=args.null_weight, excl_weight=args.excl_weight,
                                    active_weight=args.active_weight,
                                    identity_drop=args.identity_drop)
    if args.arch == "behavior":
        return BehaviorMoEIDS(**common, behavior_dim=meta.get("behavior_dim", 13),
                              null_weight=args.null_weight, excl_weight=args.excl_weight,
                              active_weight=args.active_weight,
                              identity_drop=args.identity_drop)
    return MoEIVNIDSv2(**common)


def acquire_lock(save_dir):
    """Refuse to start if another live process is using this save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    lock = os.path.join(save_dir, ".running.lock")
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            sys.exit(f"[Lock] save_dir is in use by PID {pid}: {save_dir}")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))


def per_class_f1(y_true, y_pred, classes):
    """One-vs-rest F1 per class over the full prediction set."""
    out = {}
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        d = 2 * tp + fp + fn
        out[c] = 2 * tp / d if d > 0 else 0.0
    return out


def filter_by_labels(windows, allowed):
    return [w for w in windows if w["label"] in allowed]


def split_dos_windows(windows, part):
    """Split dos (label=1) windows into two temporal halves."""
    dos = [w for w in windows if w["label"] == 1]
    others = [w for w in windows if w["label"] != 1]
    mid = len(dos) // 2
    if part == 1:
        return others + dos[:mid]
    if part == 2:
        return dos[mid:]
    return windows


def label_counts(windows):
    return np.bincount([w["label"] for w in windows], minlength=NC)


def tqdm(it, **kw):
    # no progress bars when output goes to a file (nohup)
    return _tqdm(it, disable=not sys.stderr.isatty(), mininterval=2, **kw)


# ============================================================
# Data path
# ============================================================

_PACKS = []


def make_loader(windows, args, shuffle=False, drop_last=False):
    """GPU batches when the windows belong to a packed split, DataLoader otherwise."""
    for pack in _PACKS:
        rows = pack.rows_for(windows)
        if rows is not None:
            return GPUBatches(pack, rows, args.batch_size, shuffle=shuffle, drop_last=drop_last)
    return DataLoader(WindowDataset(windows), args.batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=args.num_workers,
                      pin_memory=True, persistent_workers=args.num_workers > 0,
                      drop_last=drop_last)


def forward_batch(model, batch, device):
    if getattr(model, "needs_behavior", False):
        if "xb" not in batch:
            raise RuntimeError("behavior architectures need data from preprocess_behavior.py "
                               "or preprocess_window.py and the fast data path")
        kw = {"xb": batch["xb"], "node_atk": batch.get("node_atk")}
        if hasattr(model, "window_dim"):
            if "wb" not in batch:
                raise RuntimeError("--arch window needs data from preprocess_window.py")
            kw["wb"] = batch["wb"]
        return model.forward_padded(batch["x"], batch["ids"], batch["adj"], batch["n"], **kw)
    if "x" in batch:
        if hasattr(model, "forward_padded"):
            return model.forward_padded(batch["x"], batch["ids"], batch["adj"], batch["n"])
        return model(*lists_from_padded(batch))
    xl = [x.to(device, non_blocking=True) for x in batch["x_list"]]
    al = [a.to(device, non_blocking=True) for a in batch["adj_list"]]
    il = [i.to(device, non_blocking=True) for i in batch["id_idx_list"]]
    return model(xl, al, il)


_ATTRACT = {"labels": None, "weight": 0.0, "share": 0.0}   # set while training an expansion


def residual_attraction(out, y):
    """Hinge: attacked nodes of the new classes give the newest expert >= share of mass."""
    lab = _ATTRACT["labels"]
    if lab is None or _ATTRACT["weight"] <= 0 or "node_probs" not in out or "node_atk" not in out:
        return None
    ny = y[out["node_graph"]]
    sel = (out["node_atk"] > 0) & torch.isin(ny, lab)
    if not sel.any():
        return None
    p_new = out["node_probs"][sel, -1]
    return _ATTRACT["weight"] * F.relu(_ATTRACT["share"] - p_new).mean()


def total_loss(model, out, y, criterion, logit_adj=None):
    """CE (+ optional logit adjustment) + balance + diversity, without .item() syncs."""
    logits = out["logits"]
    if logit_adj is not None:
        logits = logits + logit_adj.to(logits.dtype)
    loss = criterion(logits, y)
    bw = getattr(model, "balance_weight", 0.0)
    if bw and "aux_loss" in out:
        loss = loss + out["aux_loss"] * bw
    dw = getattr(model, "diversity_weight", 0.0)
    if dw and "diversity_loss" in out:
        loss = loss + out["diversity_loss"] * dw
    extra = getattr(model, "extra_loss", None)
    if extra is not None:
        loss = loss + extra(out, y)
    attr = residual_attraction(out, y)
    if attr is not None:
        loss = loss + attr
    return loss


# ============================================================
# Train / eval
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, args,
                old_model=None, distill_new_labels=None, logit_adj=None, post_step=None):
    """One epoch. With old_model: CE + cosine feature distillation."""
    model.train()
    if old_model is not None:
        old_model.eval()
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    params = [p for p in model.parameters() if p.requires_grad]
    acc = torch.zeros((), device=device)
    for batch in tqdm(loader, desc="    Train", leave=False):
        y = batch["y"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = forward_batch(model, batch, device)
            loss = total_loss(model, out, y, criterion, logit_adj)
            if old_model is not None:
                with torch.no_grad():
                    old_feat = forward_batch(old_model, batch, device)["features"]
                per = 1 - F.cosine_similarity(out["features"], old_feat, dim=-1)
                if distill_new_labels is not None:
                    keep = ~torch.isin(y, distill_new_labels)
                    per = per * keep
                    dist = per.sum() / keep.sum().clamp(min=1)
                else:
                    dist = per.mean()
                loss = loss + args.dist_weight * dist
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if post_step is not None:
            post_step()
        acc += loss.detach()
    return acc.item() / max(len(loader), 1)


@torch.no_grad()
def predict(model, loader, device, args):
    model.eval()
    use_amp = device.type == "cuda" and args.amp
    logits, ys, ent, cls_ent = [], [], [], []
    nodes = {"top1": [], "top1_real": [], "probs": [], "atk": [], "ny": [], "ids": []}
    for batch in tqdm(loader, desc="    Eval", leave=False):
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = forward_batch(model, batch, device)
        y = batch["y"].to(device)
        logits.append(out["logits"].float())
        ys.append(y)
        ent.append(out["entropy"].float())
        cls_ent.append(out["cls_entropy"].float())
        if "node_probs" in out and "node_atk" in batch:
            pr = out["node_probs"].float()
            N = batch["x"].size(1)
            mask = torch.arange(N, device=device)[None, :] < batch["n"][:, None]
            nodes["top1"].append(pr.argmax(1).to(torch.int16))
            nodes["top1_real"].append((pr[:, 1:].argmax(1) + 1).to(torch.int16))
            nodes["probs"].append(pr.half())
            nodes["atk"].append(batch["node_atk"][mask].half())
            nodes["ny"].append(y[out["node_graph"]].to(torch.int16))
            nodes["ids"].append(batch["ids"][mask].to(torch.int32))
    res = {k: torch.cat(v).cpu().numpy()
           for k, v in (("logits", logits), ("y", ys), ("entropy", ent), ("cls_entropy", cls_ent))}
    if nodes["top1"]:
        res["nodes"] = {k: torch.cat(v).cpu().numpy() for k, v in nodes.items()}
    return res


def _nmi(a, b):
    """Normalised mutual information (arithmetic mean normalisation)."""
    if len(a) == 0:
        return float("nan")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    c = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(c, (ai, bi), 1)
    p = c / c.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(2 * mi / (ha + hb)) if ha + hb > 0 else 0.0


def specialization(nodes, n_experts):
    """How the behavior router uses its experts, measured with node-level ground truth.

    null_rate_benign_nodes    share of nodes without attack frames whose top choice is null
    null_rate_attacked_nodes  same for nodes with attack frames (should be LOW)
    nmi_expert_class          NMI between top choice (null included) and class, attacked nodes
    nmi_active_class          NMI between top real expert and class, attacked nodes that
                              are actually routed to a real expert
    expert_mass               per class, mean routing probability [null, e1, ..., eM]
                              over its attacked nodes (actual usage, not an argmax)
    """
    top1, real, probs, atk, ny, ids = (nodes[k] for k in
                                       ("top1", "top1_real", "probs", "atk", "ny", "ids"))
    probs = probs.astype(np.float32)
    attacked = atk > 0
    benign = ~attacked
    active = attacked & (top1 != 0)
    out = {
        "null_rate_benign_nodes": float((top1[benign] == 0).mean()) if benign.any() else None,
        "null_rate_attacked_nodes": float((top1[attacked] == 0).mean()) if attacked.any() else None,
        "nmi_expert_class": _nmi(top1[attacked], ny[attacked]),
        "nmi_active_class": _nmi(real[active], ny[active]),
        "nmi_active_canid": _nmi(real[active], ids[active]),
        "expert_mass": {},
    }
    for c in np.unique(ny[attacked]):
        sel = attacked & (ny == c)
        out["expert_mass"][CLASS_NAMES[c]] = [round(float(v), 3) for v in probs[sel].mean(0)]
    return out


HARD_CONFUSIONS = [("fuzzing", "systematic"), ("systematic", "fuzzing"),
                   ("rpm", "standstill"), ("standstill", "rpm"),
                   ("combined", "gear"), ("combined", "rpm"), ("combined", "speed"),
                   ("combined", "normal"), ("speed", "normal")]


def hard_confusions(pred_out, seen):
    """Counts of true -> predicted for the hard pairs, plus where normal false alarms go."""
    y, p = pred_out["y"], pred_out["logits"].argmax(1)
    idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    pairs = {}
    for a, b in HARD_CONFUSIONS:
        ia, ib = idx[a], idx[b]
        if ia in seen and ib in seen and (y == ia).any():
            pairs[f"{a}->{b}"] = int(((y == ia) & (p == ib)).sum())
    fa = p[(y == 0) & (p != 0)]
    top = {CLASS_NAMES[c]: int((fa == c).sum()) for c in np.unique(fa)}
    top = dict(sorted(top.items(), key=lambda kv: -kv[1])[:3])
    return {"pairs": pairs, "normal_false_alarms": top}


def metrics_from(pred_out, seen):
    y, p = pred_out["y"], pred_out["logits"].argmax(1)
    m = compute_metrics(y, p)
    present = [c for c in sorted(seen) if (y == c).any()]
    f1 = per_class_f1(y, p, present)
    normal = y == 0
    m["normal_fpr"] = float((p[normal] != 0).mean()) if normal.any() else 0.0
    m["detection_rate"] = float((p[~normal] != 0).mean()) if (~normal).any() else 0.0
    return m, {CLASS_NAMES[c]: f1[c] for c in present}


def evaluate(model, loader, device, args, seen):
    return metrics_from(predict(model, loader, device, args), seen)


def fit_phase(model, train_data, val_set, seen, phase_name, epochs, lr, args, device,
              logger, criterion, old_model=None, distill_new_labels=None,
              logit_adj=None, post_step=None, scheduler="const",
              slow_params=None, slow_mult=1.0):
    """Train with val-based checkpoint selection; returns the best model state loaded.
    slow_params: parameters trained at lr * slow_mult (e.g. old experts)."""
    train_loader = make_loader(train_data, args, shuffle=True, drop_last=True)
    val_loader = make_loader(val_set, args, shuffle=False)
    slow_ids = {id(p) for p in (slow_params or [])}
    fast = [p for p in model.parameters() if p.requires_grad and id(p) not in slow_ids]
    slow = [p for p in model.parameters() if p.requires_grad and id(p) in slow_ids]
    groups = [{"params": fast, "lr": lr}]
    if slow:
        groups.append({"params": slow, "lr": lr * slow_mult})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=args.weight_decay)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-5)
           if scheduler == "cosine" else None)
    ckpt_path = os.path.join(args.save_dir, f"{phase_name}_best.pth")
    best, log_every = -1.0, max(1, epochs // 5)
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, opt, criterion, device, args,
                           old_model, distill_new_labels, logit_adj, post_step)
        if sch is not None:
            sch.step()
        if ep % args.eval_every == 0 or ep == epochs:
            val_m, _ = evaluate(model, val_loader, device, args, seen)
            if val_m["macro_f1"] > best:
                best = val_m["macro_f1"]
                save_ckpt({"model": model.state_dict(), "args": vars(args), "phase": phase_name,
                           "num_experts": len(model.moe.experts) if hasattr(model, "moe") else 0},
                          ckpt_path)
            if ep % log_every == 0 or ep == epochs:
                logger.info(f"  Ep {ep} | loss={loss:.4f} val_F1={val_m['macro_f1']:.4f}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    return best


# ============================================================
# Replay
# ============================================================

class ReplayMemory:
    """What the learner may revisit from old phases.

    ratio / class_balanced / all draw from the full pool of past training
    windows (unbounded memory: diagnostic settings). buffer keeps at most
    K windows per class (realistic setting)."""

    def __init__(self, args):
        self.args = args
        self.pool = []                      # all past windows (not used by 'buffer')
        self.buffer = {}                    # class -> list of windows ('buffer' only)
        self.natural = np.zeros(NC)         # label counts of everything seen so far

    def add(self, windows, phase_idx):
        self.natural += label_counts(windows)
        self.pool.extend(windows)
        if self.args.replay == "buffer":
            rng = np.random.default_rng(self.args.seed + 1000 + phase_idx)
            K = self.args.buffer_per_class
            for c in np.unique([w["label"] for w in windows]):
                cand = self.buffer.get(c, []) + [w for w in windows if w["label"] == c]
                if len(cand) > K:
                    idx = rng.choice(len(cand), size=K, replace=False)
                    cand = [cand[i] for i in idx]
                self.buffer[c] = cand

    def sample(self, n_new):
        a = self.args
        if a.replay == "all":
            return list(self.pool)
        if a.replay == "buffer":
            return [w for ws in self.buffer.values() for w in ws]
        size = min(len(self.pool), int(round(a.replay_ratio * n_new)))
        if size <= 0:
            return []
        rng = np.random.default_rng(a.seed)          # same draw as earlier runs
        if a.replay == "ratio":
            idx = rng.choice(len(self.pool), size=size, replace=False)
            return [self.pool[i] for i in idx]
        # class_balanced: equal quota per old class, leftovers to classes with spare data
        by_c = {}
        for i, w in enumerate(self.pool):
            by_c.setdefault(w["label"], []).append(i)
        classes = sorted(by_c)
        quota = {c: 0 for c in classes}
        left = size
        while left > 0:
            open_c = [c for c in classes if quota[c] < len(by_c[c])]
            if not open_c:
                break
            share = max(1, left // len(open_c))
            for c in open_c:
                add = min(share, len(by_c[c]) - quota[c], left)
                quota[c] += add
                left -= add
                if left == 0:
                    break
        out = []
        for c in classes:
            if quota[c]:
                out += [self.pool[i] for i in rng.choice(by_c[c], size=quota[c], replace=False)]
        return out


class BoundedMemory:
    """Fixed-size replay memory of M windows (realistic setting).

    After every phase the memory is rebuilt from what it already holds plus the
    phase's new windows (attack windows and, under --normal_protocol recording,
    the normal windows of the new recording). Old windows that were evicted are
    gone for good. Policies:

      reservoir         uniform reservoir sampling over the stream (ER)
      class_balanced    equal quota per class, normal counted as one class (CBRS-like)
      benign_anchored   normal gets a fixed share (--anchor_share), attack classes
                        share the rest equally (in the spirit of benign-anchored CBRS)
      prior             quotas follow the natural class frequencies of the stream
                        (deployment prior), with a floor per attack class
                        (--min_per_class_frac x M)

    --stratify_normal splits the normal quota equally across recordings, so every
    driving context seen so far keeps hard negatives in memory.
    full_retrain still uses the complete pool (upper bound)."""

    def __init__(self, args):
        self.args = args
        self.M = args.memory_size
        self.pool = []                       # only for full_retrain
        self.natural = np.zeros(NC)
        self.buffer = []
        self.n_seen = 0

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _waterfill(desired, avail, total):
        """Integer quotas close to `desired`, capped by `avail`, summing to <= total."""
        quota = {c: 0 for c in desired}
        open_c = {c for c in desired if avail[c] > 0}
        left = total
        while left > 0 and open_c:
            w = np.array([max(desired[c], 1e-9) for c in open_c])
            share = dict(zip(open_c, np.floor(left * w / w.sum()).astype(int)))
            if sum(share.values()) == 0:                # spread the remainder one by one
                for c in sorted(open_c, key=lambda c: -desired[c]):
                    if left == 0:
                        break
                    if quota[c] < avail[c]:
                        quota[c] += 1
                        left -= 1
                break
            for c in list(open_c):
                add = min(share[c], avail[c] - quota[c])
                quota[c] += add
                left -= add
                if quota[c] >= avail[c]:
                    open_c.discard(c)
        return quota

    def _sample(self, items, k, rng):
        if k >= len(items):
            return list(items)
        idx = rng.choice(len(items), size=k, replace=False)
        return [items[i] for i in idx]

    def _sample_normal(self, items, k, rng):
        if not self.args.stratify_normal or k >= len(items):
            return self._sample(items, k, rng)
        by_rec = {}
        for w in items:
            by_rec.setdefault(w.get("source_file", "?"), []).append(w)
        quota = self._waterfill({r: 1.0 for r in by_rec}, {r: len(v) for r, v in by_rec.items()}, k)
        out = []
        for r, v in by_rec.items():
            out += self._sample(v, quota[r], rng)
        return out

    # -- interface ---------------------------------------------------------------
    def add(self, windows, phase_idx):
        a = self.args
        self.natural += label_counts(windows)
        self.pool.extend(windows)
        rng = np.random.default_rng(a.seed + 1000 + phase_idx)
        if a.memory_policy == "reservoir":
            for i in rng.permutation(len(windows)):
                w = windows[i]
                self.n_seen += 1
                if len(self.buffer) < self.M:
                    self.buffer.append(w)
                else:
                    j = int(rng.integers(0, self.n_seen))
                    if j < self.M:
                        self.buffer[j] = w
            return
        cand = {}
        for w in self.buffer + list(windows):
            cand.setdefault(w["label"], []).append(w)
        avail = {c: len(v) for c, v in cand.items()}
        classes = sorted(cand)
        attacks = [c for c in classes if c != 0]
        if a.memory_policy == "class_balanced":
            desired = {c: 1.0 for c in classes}
        elif a.memory_policy == "benign_anchored":
            share = a.anchor_share if 0 in cand else 0.0
            desired = {c: (share if c == 0 else (1 - share) / max(len(attacks), 1)) for c in classes}
        else:   # prior
            nat = {c: self.natural[c] for c in classes}
            tot = sum(nat.values())
            floor = a.min_per_class_frac
            desired = {c: max(nat[c] / tot, floor if c != 0 else 0.0) for c in classes}
        quota = self._waterfill(desired, avail, self.M)
        new_buf = []
        for c in classes:
            pick = self._sample_normal if c == 0 else self._sample
            new_buf += pick(cand[c], quota[c], rng)
        self.buffer = new_buf

    def sample(self, n_new):
        return list(self.buffer)

    def composition(self):
        cnt = label_counts(self.buffer)
        return {CLASS_NAMES[c]: int(cnt[c]) for c in range(NC) if cnt[c]}


def select_windows(windows, labels, recordings, protocol):
    """Attack windows of the given labels; normal windows of the given recordings
    (protocol 'recording') or all normal windows (protocol 'all', previous behaviour)."""
    if protocol == "all":
        return filter_by_labels(windows, labels)
    return [w for w in windows
            if (w["label"] != 0 and w["label"] in labels)
            or (w["label"] == 0 and w.get("source_file") in recordings)]


def logit_adjustment(train_data, natural_counts, seen, tau, device):
    """tau * (log pi_train - log pi_natural) on seen classes, 0 elsewhere.
    Added to the logits inside the loss only, so inference uses raw logits."""
    if tau <= 0:
        return None
    seen = sorted(seen)
    tr = label_counts(train_data)[seen] + 1.0
    nat = natural_counts[seen] + 1.0
    adj = np.zeros(NC, dtype=np.float32)
    adj[seen] = tau * (np.log(tr / tr.sum()) - np.log(nat / nat.sum()))
    return torch.tensor(adj, device=device)


# ============================================================
# One continual phase (expand / expand_auto)
# ============================================================

def behavior_uncertainty(nodes):
    """Routing uncertainty on nodes the router treats as abnormal (p_null < 0.5).
    Returns normalised entropy over the real experts (0 = one expert, 1 = uniform),
    the mean top-expert confidence, and the share of nodes that are active."""
    probs = nodes["probs"].astype(np.float32)
    active = probs[:, 0] < 0.5
    if active.sum() == 0 or probs.shape[1] < 3:
        return {"norm_entropy": 0.0, "confidence": 1.0, "active_frac": float(active.mean()),
                "n_active": int(active.sum())}
    real = probs[active, 1:]
    real = real / np.clip(real.sum(1, keepdims=True), 1e-12, None)
    ent = -(real * np.log(real + 1e-12)).sum(1) / np.log(real.shape[1])
    return {"norm_entropy": float(ent.mean()), "confidence": float(real.max(1).mean()),
            "active_frac": float(active.mean()), "n_active": int(active.sum())}


def decide_action(model, train_new, args, device, ent_thr, adapt_thr, logger,
                  route_base=None, behav_base=None):
    """ignore / reuse / expand from class novelty (cls entropy) and behavior novelty."""
    pre = predict(model, make_loader(train_new, args), device, args)
    ce = pre["cls_entropy"]
    cls_mean = float(ce.mean())
    cls_ratio = cls_mean / adapt_thr * args.adapt_mult if adapt_thr > 0 else float("inf")
    cls_novel = cls_mean > adapt_thr
    info = {"cls_mean": cls_mean, "cls_ratio": cls_ratio, "cls_novel": bool(cls_novel),
            "high_ratio": float((ce > ent_thr).mean())}
    logger.info(f"  class novelty   : cls entropy {cls_mean:.4f} = {cls_ratio:.1f}x baseline "
                f"({'NOVEL' if cls_novel else 'known'}, threshold {args.adapt_mult:g}x)")

    beh_novel = None
    if route_base:
        info["route_ratio"] = float(pre["entropy"].mean()) / route_base
    if behav_base is not None and "nodes" in pre:
        b = behavior_uncertainty(pre["nodes"])
        ratio = b["norm_entropy"] / max(behav_base["norm_entropy"], 1e-6)
        beh_novel = ratio > args.behav_mult
        info.update({"behav_" + k: v for k, v in b.items()})
        info.update({"behav_ratio": ratio, "behav_novel": bool(beh_novel)})
        logger.info(f"  behavior novelty: routing entropy on active nodes {b['norm_entropy']:.3f} "
                    f"= {ratio:.2f}x baseline ({'NOVEL' if beh_novel else 'known'}, threshold "
                    f"{args.behav_mult:g}x) | confidence {b['confidence']:.3f} | "
                    f"active nodes {b['active_frac']:.1%} | old route ratio "
                    f"{info.get('route_ratio', float('nan')):.2f}x")

    if args.mode == "expand":
        action = "expand"
    elif args.decision == "dual":
        if beh_novel is None:
            raise RuntimeError("--decision dual needs the behavior architecture and data")
        action = "ignore" if not cls_novel else ("expand" if beh_novel else "reuse")
    else:
        action = "expand" if cls_novel else "ignore"
    info["cls_only_would"] = "expand" if cls_novel else "ignore"
    info["decision"] = action
    logger.info(f"  >>> Decision: {action.upper()} (cls-only rule would: "
                f"{info['cls_only_would'].upper()})")
    return action, info


def adapt_and_train(model, memory, train_new, val_set, seen, phase_name, args, device,
                    logger, criterion, action="expand", context=None):
    """action expand: add a residual expert; action reuse: keep experts and router,
    train the classifier head on the reused expert outputs."""
    # teacher for feature distillation, taken before the model changes
    old_model = None
    if args.dist_weight > 0:
        old_model = copy.deepcopy(model).eval()
        for p in old_model.parameters():
            p.requires_grad = False

    new_label_t = torch.tensor(sorted({w["label"] for w in train_new}), device=device)
    if action == "expand":
        model.expand_expert_pool(device=device)
        model.freeze_for_expansion()
    else:   # reuse
        for p in model.parameters():
            p.requires_grad = False
        for mod in (model.classifier, model.cos_head, model.readout_norm):
            for p in mod.parameters():
                p.requires_grad = True
    if args.train_scope == "all":
        model.unfreeze_all()
    elif args.train_scope == "backbone":
        # experts (and, with --freeze_old_gate, the old router rows) stay frozen to keep
        # the attack-behavior knowledge; the identity path, i.e. the model of normal
        # traffic per CAN ID, is allowed to adapt (feature distillation still applies)
        if not hasattr(model, "backbone"):
            sys.exit("--train_scope backbone needs the behavior/window architecture")
        for mod in (model.id_embedding, model.backbone, model.post_norm):
            for p in mod.parameters():
                p.requires_grad = True

    slow_params = None
    if args.old_lr_mult > 0 and args.train_scope != "all" and hasattr(model, "moe"):
        old_experts = model.moe.experts[:-1] if action == "expand" else model.moe.experts
        slow_params = [p for e in old_experts for p in e.parameters()]
        for p in slow_params:
            p.requires_grad = True

    post_step = None
    if args.freeze_old_gate and action == "expand":
        gate = model.moe.router.gate
        pinned = [p for p in (gate.weight, gate.bias) if p is not None]
        n_old = gate.weight.size(0) - 1     # every row except the new expert's
        saved = [p[:n_old].detach().clone() for p in pinned]

        def _mask(g):
            g = g.clone()
            g[:n_old] = 0
            return g
        for p in pinned:
            p.register_hook(_mask)

        def post_step():                    # undo AdamW weight decay on pinned rows
            with torch.no_grad():
                for p, v in zip(pinned, saved):
                    p[:n_old].copy_(v)

    if getattr(model, "needs_behavior", False) and action == "expand":
        model.excl_new_labels = new_label_t
        _ATTRACT.update(labels=new_label_t, weight=args.attract_weight,
                        share=args.attract_share)

    saved_bw = getattr(model, "balance_weight", None)
    if args.no_balance_on_expand and saved_bw is not None:
        model.balance_weight = 0.0

    replay = memory.sample(len(train_new))       # replay size still set by the attack windows
    ctx = []
    if context:
        in_replay = {id(w) for w in replay}
        ctx = [w for w in context if id(w) not in in_replay]
    train_data = train_new + replay + ctx
    natural = memory.natural + label_counts(train_new)
    if args.normal_protocol == "recording" and ctx:
        natural = natural + label_counts(ctx)          # new normals are not in memory yet
    adj = logit_adjustment(train_data, natural,
                           seen, args.logit_adjust, device)
    new_labels = None
    if old_model is not None and args.distill_on == "old":
        new_labels = torch.tensor(sorted({w["label"] for w in train_new}), device=device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if isinstance(memory, BoundedMemory):
        logger.info(f"  Memory ({args.memory_policy}, M={args.memory_size}"
                    f"{', stratified normal' if args.stratify_normal else ''}): {memory.composition()}")
    logger.info(f"  Action={action} | attract={args.attract_weight if action == 'expand' else 0}"
                f"@{args.attract_share} | old_lr_mult={args.old_lr_mult}")
    logger.info(f"  Train: new={len(train_new)} replay={len(replay)} ({args.replay}) "
                f"context_normal={len(ctx)} | "
                f"trainable params={trainable:,} | distill={args.dist_weight if old_model is not None else 0}"
                f"/{args.distill_on} | logit_adjust={args.logit_adjust} | "
                f"freeze_old_gate={args.freeze_old_gate} | balance={'off' if args.no_balance_on_expand else 'on'}")
    if adj is not None:
        logger.info("  Logit adjustment: " + ", ".join(
            f"{CLASS_NAMES[c]}={adj[c].item():+.2f}" for c in sorted(seen)))

    fit_phase(model, train_data, val_set, seen, phase_name, args.phase_epochs,
              args.lr * args.phase_lr_mult, args, device, logger, criterion,
              old_model=old_model, distill_new_labels=new_labels, logit_adj=adj,
              post_step=post_step, scheduler=args.phase_scheduler,
              slow_params=slow_params, slow_mult=args.old_lr_mult)

    _ATTRACT.update(labels=None, weight=0.0, share=0.0)
    if saved_bw is not None:
        model.balance_weight = saved_bw
    if getattr(model, "needs_behavior", False):
        model.excl_new_labels = None
    del old_model


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("MoE-IVN-IDS continual learning (final)")
    g = p.add_argument_group("run")
    g.add_argument("--data_dir", default="./data/2011-chevrolet-impala/preprocessed")
    g.add_argument("--save_dir", required=True)
    g.add_argument("--mode", default="expand_auto",
                   choices=["expand_auto", "expand", "fine_tune", "full_retrain"])
    g.add_argument("--phase_config", default="variant", choices=["default", "variant"])
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--resume_phase1", default=None,
                   help="phase1_best.pth to reuse; skips base training")
    g.add_argument("--run_name", default=None,
                   help="name used in the log file (default: last folder of --save_dir)")

    g = p.add_argument_group("hardware")
    g.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    g.add_argument("--cpu_threads", type=int, default=4,
                   help="cap for torch/BLAS CPU threads")
    g.add_argument("--fast_data", default="auto", choices=["auto", "gpu", "cpu", "off"])
    g.add_argument("--num_workers", type=int, default=2,
                   help="only used for one-time packing or with --fast_data off")
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--eval_every", type=int, default=1,
                   help="validate every N epochs (checkpoint selection granularity)")
    g.add_argument("--pack_only", action="store_true",
                   help="build the packed data cache and exit (run once before parallel jobs)")
    g.add_argument("--save_preds", action=argparse.BooleanOptionalAction, default=True,
                   help="save test logits/labels per phase to preds_<phase>.npz")

    g = p.add_argument_group("model")
    g.add_argument("--arch", default="cosine", choices=["cosine", "v2", "behavior", "window"])
    g.add_argument("--null_weight", type=float, default=0.1,
                   help="behavior: push nodes of normal windows to the null expert")
    g.add_argument("--excl_weight", type=float, default=0.1,
                   help="behavior: keep old-class nodes off the new expert during expansion")
    g.add_argument("--active_weight", type=float, default=0.5,
                   help="behavior: push nodes with attack frames away from the null expert")
    g.add_argument("--identity_drop", type=float, default=0.3,
                   help="behavior: per-window dropout of the identity readout during training")
    g.add_argument("--num_experts", type=int, default=6)
    g.add_argument("--top_k", type=int, default=2)
    g.add_argument("--embed_dim", type=int, default=16)
    g.add_argument("--moe_out_dim", type=int, default=128)
    g.add_argument("--dropout", type=float, default=0.2)

    g = p.add_argument_group("optimisation")
    g.add_argument("--batch_size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=1e-4)
    g.add_argument("--phase1_epochs", type=int, default=100)
    g.add_argument("--phase_epochs", type=int, default=30)
    g.add_argument("--phase_lr_mult", type=float, default=0.5,
                   help="lr after expansion = lr * phase_lr_mult")
    g.add_argument("--phase_scheduler", default="const", choices=["const", "cosine"])
    g.add_argument("--retrain_epochs", type=int, default=None,
                   help="epochs per phase for full_retrain (default: phase_epochs)")

    g = p.add_argument_group("expansion decision")
    g.add_argument("--decision_rule", default="mean", choices=["mean", "or"])
    g.add_argument("--decision", default="dual", choices=["dual", "cls"],
                   help="dual: ignore/reuse/expand from class + behavior novelty; "
                        "cls: expand whenever the class is novel (previous behaviour)")
    g.add_argument("--behav_mult", type=float, default=1.5,
                   help="behavior novel when routing entropy on active nodes > mult x baseline")
    g.add_argument("--attract_weight", type=float, default=1.0,
                   help="residual expansion: weight of the new-expert attraction hinge (0 = off)")
    g.add_argument("--attract_share", type=float, default=0.3,
                   help="residual expansion: minimum routing mass on the new expert for "
                        "attacked nodes of the new classes")
    g.add_argument("--reuse_tol", type=float, default=0.01,
                   help="reuse is rejected (-> residual expansion) if an old class loses more "
                        "than this F1 on val, or the new class stays below --reuse_min_f1; "
                        "-1 disables the check")
    g.add_argument("--reuse_min_f1", type=float, default=0.95)
    g.add_argument("--old_lr_mult", type=float, default=0.0,
                   help="0: old experts frozen; >0: trainable at lr * old_lr_mult")
    g.add_argument("--novelty", default="cls", choices=["cls", "route"],
                   help="cls: classification entropy; route: behavior routing entropy "
                        "(behavior arch only). The route ratio is always logged when available")
    g.add_argument("--route_mult", type=float, default=2.0,
                   help="--novelty route: expand when mean route entropy > route_mult x baseline")
    g.add_argument("--adapt_mult", type=float, default=10.0)
    g.add_argument("--entropy_percentile", type=int, default=95)

    g = p.add_argument_group("after expansion")
    g.add_argument("--dist_weight", type=float, default=2.0)
    g.add_argument("--distill_on", default="all", choices=["all", "old"])
    g.add_argument("--train_scope", default="default", choices=["default", "all", "backbone"],
                   help="default: model.freeze_for_expansion(); all: everything trainable; "
                        "backbone: experts frozen, identity path (embedding + backbone) trainable")
    g.add_argument("--freeze_old_gate", action="store_true",
                   help="keep router rows of existing experts fixed")
    g.add_argument("--no_balance_on_expand", action="store_true",
                   help="drop the load-balance loss while training after expansion")
    g.add_argument("--logit_adjust", type=float, default=0.0,
                   help="tau for logit-adjusted CE after expansion (0 = off)")

    g = p.add_argument_group("replay")
    g.add_argument("--normal_protocol", default="all", choices=["all", "recording"],
                   help="all: every normal window is available from phase 1 (previous runs); "
                        "recording: normal windows arrive with their recording, i.e. phase 1 "
                        "only sees the normals of the base recordings and each later phase "
                        "brings the normals of its own recording (implies context normal)")
    g.add_argument("--new_normal", default="train", choices=["train", "skip"],
                   help="recording protocol: train on the normal windows of a new recording "
                        "(default) or drop them (ablation: new data = attack windows only)")
    g.add_argument("--memory_size", type=int, default=0,
                   help="fixed replay memory of M windows (0 = previous unbounded --replay modes)")
    g.add_argument("--memory_policy", default="prior",
                   choices=["reservoir", "class_balanced", "benign_anchored", "prior"])
    g.add_argument("--min_per_class_frac", type=float, default=0.02,
                   help="prior policy: floor per attack class, as a fraction of M")
    g.add_argument("--anchor_share", type=float, default=0.5,
                   help="benign_anchored policy: share of the memory reserved for normal")
    g.add_argument("--stratify_normal", action="store_true",
                   help="split the normal part of the memory equally across recordings")
    g.add_argument("--context_normal", action="store_true",
                   help="add the normal windows of the new attack's recording(s) to the phase")
    g.add_argument("--context_normal_max", type=int, default=0,
                   help="cap on context normal windows per phase (0 = all)")
    g.add_argument("--replay", default="ratio",
                   choices=["ratio", "class_balanced", "buffer", "all"])
    g.add_argument("--replay_ratio", type=float, default=2.0,
                   help="replay size = ratio x new data (ratio, class_balanced)")
    g.add_argument("--buffer_per_class", type=int, default=200,
                   help="memory per class for --replay buffer")
    a = p.parse_args()
    if a.arch == "v2" and a.dist_weight > 0 and a.mode in ("expand", "expand_auto"):
        p.error("arch v2 returns no 'features'; use --dist_weight 0")
    if a.eval_every < 1:
        p.error("--eval_every must be >= 1")
    if a.arch in ("behavior", "window") and a.fast_data == "off":
        p.error("arch behavior needs the fast data path")
    return a


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            sys.exit(f"--gpu {args.gpu} but only {torch.cuda.device_count()} CUDA devices visible")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    acquire_lock(args.save_dir)
    args.run_name = args.run_name or Path(args.save_dir).resolve().name
    args.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    args.log_name = f"{args.run_name}_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(args.save_dir, args.log_name)
    logger.info(f"Run: {args.run_name} | started {args.started_at} | log: {args.log_name}")
    logger.info(f"Mode: {args.mode} | Phase config: {args.phase_config} | device: {device}"
                + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    logger.info(json.dumps(vars(args), indent=2))

    data = Path(args.data_dir)
    meta = load_metadata(data / "metadata.json")
    all_train = load_windows(data / "windows_train.pt")
    all_val = load_windows(data / "windows_val.pt")
    all_test = load_windows(data / "windows_test.pt")
    if args.fast_data != "off":
        for split, ws in (("train", all_train), ("val", all_val), ("test", all_test)):
            _PACKS.append(PackedSplit(ws, data / f"packed_{split}.pt", device,
                                      src_path=data / f"windows_{split}.pt",
                                      placement=args.fast_data, num_workers=args.num_workers,
                                      log=logger.info))

    if args.pack_only:
        logger.info("Packed data cache is ready.")
        return

    # packing may build a DataLoader (which draws from the torch RNG) only on the
    # first run; reseed so model init and shuffling do not depend on the cache state
    set_seed(args.seed)

    phases = VARIANT_PHASES if args.phase_config == "variant" else DEFAULT_PHASES
    criterion = nn.CrossEntropyLoss()
    results, seen = [], set()
    memory = BoundedMemory(args) if args.memory_size > 0 else ReplayMemory(args)
    recordings = set()

    def record(phase, ptype, mode, test_out, decision, t0, model):
        m, per = metrics_from(test_out, seen)
        n_exp = len(model.moe.experts) if hasattr(model, "moe") else 0
        logger.info(f"\n  {phase['name']} result: F1={m['macro_f1']:.4f} acc={m['acc']:.4f} "
                    f"normal_FPR={m['normal_fpr']:.4f} detection={m['detection_rate']:.4f} "
                    f"experts={n_exp} | {(time.time() - t0) / 60:.1f} min")
        logger.info(f"  Per-attack: {json.dumps({k: round(v, 4) for k, v in per.items()})}")
        hc = hard_confusions(test_out, seen)
        if hc["pairs"]:
            logger.info("  Hard confusions (test, true->pred): "
                        + ", ".join(f"{k}={v}" for k, v in hc["pairs"].items()))
        logger.info(f"  Normal false alarms go to: {hc['normal_false_alarms']}")
        spec = None
        if "nodes" in test_out:
            spec = specialization(test_out["nodes"], n_exp)
            logger.info(f"  Specialization: null(benign nodes)={spec['null_rate_benign_nodes']:.3f} "
                        f"null(attacked nodes)={spec['null_rate_attacked_nodes']:.3f} "
                        f"NMI(top choice,class)={spec['nmi_expert_class']:.3f} "
                        f"NMI(active expert,class)={spec['nmi_active_class']:.3f} "
                        f"NMI(active expert,CAN ID)={spec['nmi_active_canid']:.3f}")
            logger.info("    routing mass on attacked nodes [null, e1, e2, ...]:")
            for cname, mass in spec["expert_mass"].items():
                logger.info(f"    {cname:>12s} {mass}")
        if args.save_preds:
            extra = {}
            if "nodes" in test_out:
                extra = {f"node_{k}": v for k, v in test_out["nodes"].items()}
            np.savez_compressed(os.path.join(args.save_dir, f"preds_{phase['name']}.npz"),
                                y=test_out["y"], logits=test_out["logits"].astype(np.float16),
                                route_entropy=test_out["entropy"].astype(np.float16), **extra)
        results.append({"phase": phase["name"], "desc": phase["desc"], "type": ptype,
                        "mode": mode, "overall": m, "per_attack": per, "num_experts": n_exp,
                        "decision_info": decision, "specialization": spec,
                        "hard_confusions": hc,
                        "minutes": (time.time() - t0) / 60})

    # ---------------------------------------------------------------- phase 1
    t0 = time.time()
    phase = phases[0]
    seen.update(phase["attacks"])
    logger.info(f"\n{'=' * 60}\n{phase['name']}: {phase['desc']}\n{'=' * 60}")
    recordings |= {CLASS_NAMES[c] for c in phase["attacks"] if c != 0}
    train_p1 = select_windows(all_train, seen, recordings, args.normal_protocol)
    val_p1 = select_windows(all_val, seen, recordings, args.normal_protocol)
    test_p1 = select_windows(all_test, seen, recordings, args.normal_protocol)
    if args.normal_protocol == "recording":
        logger.info(f"  Normal protocol 'recording': normals from {sorted(recordings)}")
    if args.phase_config == "variant" and phase.get("use_dos_part") == 1:
        train_p1, val_p1, test_p1 = (split_dos_windows(x, 1) for x in (train_p1, val_p1, test_p1))
    logger.info(f"  Data: train={len(train_p1)} val={len(val_p1)} test={len(test_p1)}")

    model = build_model(args, meta).to(device)
    logger.info(f"  Model: {model.get_info()}")
    p1_path = os.path.join(args.save_dir, "phase1_best.pth")
    if args.resume_phase1:
        prev = torch.load(args.resume_phase1, map_location="cpu", weights_only=False)
        for k in ("arch", "num_experts", "top_k", "embed_dim", "moe_out_dim", "seed"):
            pv = prev.get("args", {}).get(k)
            if pv is not None and pv != getattr(args, k):
                sys.exit(f"[Resume] {k} mismatch: checkpoint={pv} args={getattr(args, k)}")
        if os.path.abspath(args.resume_phase1) != os.path.abspath(p1_path):
            shutil.copy(args.resume_phase1, p1_path)
        model.load_state_dict(prev["model"])
        logger.info(f"  [Resume] loaded {args.resume_phase1}")
    else:
        opt_args = dict(args=args, device=device, logger=logger, criterion=criterion)
        fit_phase(model, train_p1, val_p1, seen, "phase1", args.phase1_epochs, args.lr,
                  scheduler="cosine", **opt_args)
        ck = torch.load(p1_path, map_location="cpu", weights_only=False)
        ck["vocab_size"] = meta["vocab_size"]
        save_ckpt(ck, p1_path)

    record(phase, "base", "initial", predict(model, make_loader(test_p1, args), device, args),
           {"decision": "initial_training"}, t0, model)
    memory.add(train_p1, 0)

    ent_thr = adapt_thr = route_base = behav_base = None
    if args.mode in ("expand", "expand_auto"):
        val_out = predict(model, make_loader(val_p1, args), device, args)
        if "nodes" in val_out:
            route_base = float(val_out["entropy"].mean())
            logger.info(f"[Threshold] baseline route entropy (max over nodes) mean={route_base:.4f}")
            behav_base = behavior_uncertainty(val_out["nodes"])
            logger.info(f"[Threshold] baseline behavior uncertainty on active nodes: "
                        f"entropy={behav_base['norm_entropy']:.3f} "
                        f"confidence={behav_base['confidence']:.3f} "
                        f"active={behav_base['active_frac']:.1%} | behav threshold "
                        f"{args.behav_mult:g}x")
        ce = val_out["cls_entropy"]
        ent_thr = float(np.percentile(ce, args.entropy_percentile))
        adapt_thr = float(ce.mean()) * args.adapt_mult
        logger.info(f"\n[Threshold] baseline cls_entropy mean={ce.mean():.4f} std={ce.std():.4f} "
                    f"p{args.entropy_percentile}={ent_thr:.4f} | adapt={adapt_thr:.6f} "
                    f"({args.adapt_mult:g}x mean)")

    # ---------------------------------------------------------------- phases 2..
    for pi, phase in enumerate(phases[1:], start=1):
        t0 = time.time()
        ptype = phase.get("type", "new")
        logger.info(f"\n{'=' * 60}\n{phase['name']}: {phase['desc']}\n  Type: {ptype}\n{'=' * 60}")
        new = set(phase["attacks"])
        seen_before = set(seen)
        train_new = filter_by_labels(all_train, new)
        if ptype == "variant" and phase.get("use_dos_part") == 2:
            train_new = split_dos_windows(train_new, 2)
        seen.update(new)
        new_recs = {CLASS_NAMES[c] for c in new if c != 0} - recordings
        recordings |= new_recs
        val_all = select_windows(all_val, seen, recordings, args.normal_protocol)
        test_all = select_windows(all_test, seen, recordings, args.normal_protocol)
        logger.info(f"  New train: {len(train_new)} | Val: {len(val_all)} | Test: {len(test_all)}")

        context = None
        if args.normal_protocol == "recording":
            # the normal traffic of a newly recorded attack arrives with it: it is new data,
            # not replay, and it joins the memory afterwards like any other new window
            context = [w for w in all_train if w["label"] == 0 and w.get("source_file") in new_recs]
            logger.info(f"  New recording(s) {sorted(new_recs) or '-'}: {len(context)} normal windows")
            if args.new_normal == "skip":
                # ablation: new data = attack windows only (usual CL-IDS assumption); the
                # normal windows of the new recording are neither trained on nor stored
                logger.info("  --new_normal skip: these normal windows are NOT used")
                context = []
        elif args.context_normal:
            # normal windows of the recording(s) the new attack comes from: in practice the
            # traffic around a newly recorded attack is captured with it
            srcs = {w.get("source_file") for w in train_new}
            context = [w for w in all_train if w["label"] == 0 and w.get("source_file") in srcs]
            if args.context_normal_max and len(context) > args.context_normal_max:
                idx = np.random.default_rng(args.seed + 7).choice(
                    len(context), size=args.context_normal_max, replace=False)
                context = [context[i] for i in idx]
            logger.info(f"  Context normal: {len(context)} normal windows from {sorted(srcs)}")
        if args.mode in ("expand", "expand_auto"):
            action, decision = decide_action(model, train_new, args, device, ent_thr,
                                             adapt_thr, logger, route_base, behav_base)
            if action == "reuse" and args.reuse_tol >= 0:
                # reuse is only safe if the new class can be told apart from the old
                # classes that share its experts; otherwise fall back to a residual expert
                val_old = filter_by_labels(all_val, seen_before)
                _, before = evaluate(model, make_loader(val_old, args), device, args, seen_before)
                snapshot = copy.deepcopy(model.state_dict())
                adapt_and_train(model, memory, train_new, val_all, seen, phase["name"],
                                args, device, logger, criterion, action="reuse",
                                context=context)
                _, after = evaluate(model, make_loader(val_all, args), device, args, seen)
                drops = {c: before[c] - after.get(c, 0.0) for c in before}
                worst = max(drops, key=drops.get)
                new_f1 = min(after.get(CLASS_NAMES[c], 0.0) for c in new)
                decision["reuse_check"] = {"worst_old_class": worst, "drop": drops[worst],
                                           "new_class_f1": new_f1}
                logger.info(f"  Reuse check (val): largest old-class drop {worst} "
                            f"{drops[worst]:+.4f}, new-class F1 {new_f1:.4f} "
                            f"(tolerance {args.reuse_tol})")
                if drops[worst] > args.reuse_tol or new_f1 < args.reuse_min_f1:
                    logger.info("  >>> Reuse REJECTED: restoring and expanding with a residual expert")
                    model.load_state_dict(snapshot)
                    adapt_and_train(model, memory, train_new, val_all, seen, phase["name"],
                                    args, device, logger, criterion, action="expand",
                                    context=context)
                    decision["decision"] = "expand_after_reuse"
            elif action in ("expand", "reuse"):
                adapt_and_train(model, memory, train_new, val_all, seen, phase["name"],
                                args, device, logger, criterion, action=action,
                                context=context)
        elif args.mode == "fine_tune":
            decision = {"decision": "fine_tune"}
            model.unfreeze_all()
            fit_phase(model, train_new, val_all, seen, phase["name"], args.phase_epochs,
                      args.lr * args.phase_lr_mult, args, device, logger, criterion,
                      scheduler=args.phase_scheduler)
        else:   # full_retrain: joint training from scratch on everything seen (upper bound)
            decision = {"decision": "full_retrain"}
            model = build_model(args, meta).to(device)
            fit_phase(model, memory.pool + train_new + (context or []), val_all, seen, phase["name"],
                      args.retrain_epochs or args.phase_epochs, args.lr, args, device,
                      logger, criterion, scheduler="cosine")

        memory.add(train_new + (context if args.normal_protocol == "recording" and context else []), pi)
        record(phase, ptype, args.mode, predict(model, make_loader(test_all, args), device, args),
               decision, t0, model)

    # ---------------------------------------------------------------- summary
    logger.info(f"\n{'=' * 60}\nSummary ({args.mode}, {args.phase_config})\n{'=' * 60}")
    for r in results:
        o = r["overall"]
        logger.info(f"  {r['phase']} | F1={o['macro_f1']:.4f} | FPR={o['normal_fpr']:.4f} | "
                    f"det={o['detection_rate']:.4f} | experts={r['num_experts']} | "
                    f"{r['decision_info'].get('decision')} | {r['minutes']:.1f} min")

    # continual-learning summary from per-class F1
    learned, forget, plastic = {}, {}, []
    for i, r in enumerate(results):
        for name, v in r["per_attack"].items():
            if name not in learned:
                learned[name] = i
                if i > 0:
                    plastic.append(v)
    final = results[-1]["per_attack"]
    for name, i0 in learned.items():
        hist = [results[j]["per_attack"][name] for j in range(i0, len(results) - 1)
                if name in results[j]["per_attack"]]
        if hist:
            forget[name] = max(0.0, max(hist) - final[name])
    cl = {"run_name": args.run_name, "started_at": args.started_at,
          "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "log_name": args.log_name,
          "final_macro_f1": results[-1]["overall"]["macro_f1"],
          "final_normal_fpr": results[-1]["overall"]["normal_fpr"],
          "final_detection_rate": results[-1]["overall"]["detection_rate"],
          "forgetting_max": max(forget.values()) if forget else 0.0,
          "forgetting_max_class": max(forget, key=forget.get) if forget else None,
          "forgetting_mean": float(np.mean(list(forget.values()))) if forget else 0.0,
          "new_class_f1_when_learned": float(np.mean(plastic)) if plastic else None,
          "total_minutes": sum(r["minutes"] for r in results)}
    logger.info("\nCL summary: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in cl.items()}))

    logger.info("\nPer-class F1 across phases:")
    for name in [c for c in CLASS_NAMES if c in learned]:
        logger.info(f"  {name:>12s}: " + " → ".join(
            f"{r['per_attack'].get(name, 0):.3f}" for r in results))

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(args.save_dir, "cl_summary.json"), "w") as f:
        json.dump(cl, f, indent=2, default=str)
    logger.info(f"\nDone. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()