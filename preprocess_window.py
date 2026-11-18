#!/usr/bin/env python3
"""
preprocess_window.py: behavior preprocessing + WINDOW-level behavior features.

--behavior_norm selects how behavior features are expressed relative to the
vehicle's own normal traffic:
  zscore           (default, previous runs) (v - mean) / std of the ID's normal values
  surprise         conformal tail surprise  s = -log10(p), with
                   p = min(1, 2 min(#{n <= v}+1, #{n >= v}+1) / (N+1))
                   over the N normal values of that ID (node features) or of all
                   normal windows (window features). Under normal traffic p is
                   uniform on every vehicle, so s has the same distribution
                   everywhere, whatever the scale or direction of the raw signal.
  surprise_signed  sign(v - normal median) * s, keeps the direction of the deviation

--payload_features adds 3 node features on payload continuity with the neighbouring
frames of the same CAN ID (jump, spike, bytes changed), aimed at signal-spoofing
attacks (gear, rpm, speed, standstill, interval). Without the flag the output is
identical to before.

Same windows, splits and fields as preprocess_behavior.py (x, xb, node_atk, ...),
plus one vector per window:

  wb [9]  window-level behavior, computed from the raw 64-message sequence and
          z-scored against NORMAL training windows. No feature carries the
          identity of a CAN ID; all are relative to the vehicle's own normal
          traffic, so they transfer across vehicles.

     0 id_inc1_ratio          frames followed within 8 frames by ID + 1         (systematic)
     1 id_inc_longest_run     longest chain of +1 increments (8-frame gaps) / 63 (systematic)
     2 unknown_id_ratio       frames with an ID never seen in normal traffic    (fuzzing, systematic)
     3 novel_transition_ratio consecutive ID pairs never seen in normal traffic (any injection)
     4 max_freq_ratio_log     log of max over IDs of count / normal count       (DoS, standstill, spoofing)
     5 n_elevated_ids         IDs with count > 1.5 x normal                     (combined vs single spoofing)
     6 min_gap_ratio_neglog   -log10 of min over IDs of (same-ID gap / normal period)   (interval)
     7 dlc_anomaly_ratio      frames of known IDs whose DLC differs from normal (fuzzing)
     8 distinct_ids           number of distinct IDs in the window

At the end a separability report (AUROC of the window features alone on hard
class pairs, on the val split) is printed and saved to window_report.json.

Behavior-preprocessing description:

Identical to preprocess.py for windows, splits and the "x" field (with the
default --x_norm_source all), so models trained on the old data stay
comparable. Adds two per-node fields:

  xb        [N, 13]  behavior features: the 12 raw features z-scored against
                     the NORMAL profile of the same CAN ID (mean/std computed
                     from normal windows of the training split only), clipped
                     to [-20, 20], plus an "unknown ID" flag (1 when the ID has
                     no normal profile; its z-scores then use the global normal
                     statistics). Normal traffic gives values near 0 whatever
                     the ID, so these features carry behavior, not identity.
  node_atk  [N]      fraction of attack frames of this CAN ID in the window
                     (ground truth for measuring expert specialisation).

Original description:

Feature set (12 per CAN ID per window):
  f0:     relative frequency (count / window_size)
  f1:     mean inter-arrival time
  f2:     std inter-arrival time
  f3-f10: byte means (8 bytes, normalized to [0,1])
  f11:    byte change rate

Also outputs:
  - CAN ID vocabulary (for learnable embedding in model)
  - Temporal co-occurrence adjacency per window
  - Per-file processing (no cross-file contamination)
  - No overlap (stride = window_size)

Usage:
    python preprocess.py \
        --input_dir  ./data/2011-chevrolet-impala/merged \
        --output_dir ./data/2011-chevrolet-impala/preprocessed_window \
        --window_size 64 --max_normal_per_file 5000
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ============================================================
# Constants
# ============================================================

ATTACK_MAP = {
    "normal": 0, "dos": 1, "fuzzing": 2, "gear": 3, "interval": 4,
    "rpm": 5, "speed": 6, "standstill": 7, "systematic": 8, "combined": 9,
}
INDEX_TO_LABEL = {v: k for k, v in ATTACK_MAP.items()}
NUM_CLASSES = len(ATTACK_MAP)
NUM_FEATURES = 12
WINDOW_FEATURES = [
    "id_inc1_ratio", "id_inc_longest_run", "unknown_id_ratio", "novel_transition_ratio",
    "max_freq_ratio_log", "n_elevated_ids", "min_gap_ratio_neglog", "dlc_anomaly_ratio",
    "distinct_ids",
]
NID = 4096          # covers 11-bit CAN IDs (0x000 to 0x7FF) with margin
INC_LAG = 8         # look-ahead (frames) when following +1 ID increments


# ============================================================
# Payload parsing
# ============================================================

def parse_payloads(series: pd.Series) -> np.ndarray:
    """Hex strings → [N, 8] uint8 array."""
    def to_bytes(h):
        s = str(h).strip().lower().replace("0x", "")
        s = "".join(c for c in s if c in "0123456789abcdef")
        s = s.zfill(16)[:16]
        return [int(s[i:i+2], 16) for i in range(0, 16, 2)]
    return np.array([to_bytes(x) for x in series.values], dtype=np.uint8)


def parse_dlc(series: pd.Series) -> np.ndarray:
    """Data length in bytes, from the hex string BEFORE zero padding."""
    def n_bytes(h):
        s = str(h).strip().lower().replace("0x", "")
        s = "".join(c for c in s if c in "0123456789abcdef")
        return min(len(s) // 2, 8)
    return np.array([n_bytes(x) for x in series.values], dtype=np.int8)


PAYLOAD_FEATURES = ["payload_max_jump", "payload_max_spike", "payload_bytes_changed"]


def payload_stream_features(arb_ids: np.ndarray, byte_data: np.ndarray) -> np.ndarray:
    """Per-message payload continuity w.r.t. the neighbouring frames of the SAME CAN ID
    (whole recording, time order). Vehicle-agnostic: the per-ID normal profile later
    turns them into deviations from that ID's own normal behaviour.

      jump    max over bytes of |b_t - b_prev|                  (abrupt value change)
      spike   max over bytes of max(0, min(|b_t - b_prev|, |b_next - b_t|) - |b_next - b_prev|)
              large when frame t departs from BOTH neighbours while they agree with each
              other, i.e. an injected value between two genuine frames (uses the next
              frame of the same ID: one-period look-ahead)
      nchg    number of bytes that differ from the previous frame of the same ID
    Returns [N, 3] float32 scaled to [0, 1]."""
    n = len(arb_ids)
    order = np.lexsort((np.arange(n), arb_ids))
    ids = arb_ids[order]
    B = byte_data[order].astype(np.int16)
    jump = np.zeros(n, np.float32)
    nchg = np.zeros(n, np.float32)
    spike = np.zeros(n, np.float32)
    same = ids[1:] == ids[:-1]
    d = np.abs(B[1:] - B[:-1])                                    # [n-1, 8]
    jump[1:][same] = d[same].max(1)
    nchg[1:][same] = (d[same] > 0).sum(1)
    mid = same[:-1] & same[1:]                                    # t-1, t, t+1 share the ID
    a, b = d[:-1][mid], d[1:][mid]
    c = np.abs(B[2:] - B[:-2])[mid]
    spike[1:-1][mid] = np.clip(np.minimum(a, b) - c, 0, None).max(1)
    out = np.zeros((n, 3), np.float32)
    out[order, 0] = jump / 255.0
    out[order, 1] = spike / 255.0
    out[order, 2] = nchg / 8.0
    return out


# ============================================================
# CAN ID vocabulary
# ============================================================

def build_vocab(df: pd.DataFrame) -> Dict[int, int]:
    """Build CAN ID → index mapping. Index 0 reserved for unknown."""
    unique_ids = sorted(df["arbitration_id"].unique())
    vocab = {cid: i + 1 for i, cid in enumerate(unique_ids)}  # 0 = UNK
    return vocab


# ============================================================
# Per-window feature computation (12 features, vectorized)
# ============================================================

def compute_windows(
    df: pd.DataFrame,
    byte_data: np.ndarray,
    dlc_data: np.ndarray,
    attack_type_id: int,
    vocab: Dict[int, int],
    window_size: int,
    min_unique_ids: int,
    max_normal: int,
    seed: int,
    payload: np.ndarray = None,
) -> List[Dict]:
    W = window_size
    timestamps = df["timestamp"].values.astype(np.float64)
    arb_ids = df["arbitration_id"].values.astype(np.int64)
    attacks = df["attack"].values.astype(np.int32)
    byte_float = byte_data.astype(np.float64) / 255.0

    num_windows = max(0, (len(df) - W) // W + 1)
    windows_attack, windows_normal = [], []

    for wi in tqdm(range(num_windows), desc="  Windows", leave=False):
        start = wi * W
        end = start + W

        w_ts = timestamps[start:end]
        w_ids = arb_ids[start:end]
        w_atk = attacks[start:end]
        w_bytes = byte_data[start:end]
        w_bf = byte_float[start:end]

        unique_ids, inverse, counts = np.unique(w_ids, return_inverse=True, return_counts=True)
        N = len(unique_ids)
        if N < min_unique_ids:
            continue

        # 12 features per CAN ID
        X = np.zeros((N, NUM_FEATURES), dtype=np.float64)
        node_atk = np.zeros(N, dtype=np.float64)

        for li in range(N):
            mask = inverse == li
            K = counts[li]

            # f0: frequency
            X[li, 0] = K / W

            # f1-f2: timing
            if K >= 2:
                iat = np.diff(w_ts[mask])
                X[li, 1] = iat.mean()
                X[li, 2] = iat.std()

            # f3-f10: byte means
            X[li, 3:11] = w_bf[mask].mean(axis=0)

            # f11: byte change rate
            if K >= 2:
                X[li, 11] = (w_bytes[mask][1:] != w_bytes[mask][:-1]).astype(np.float64).mean()

            # f12-f14 (--payload_features): continuity with the neighbouring frames of
            # the same ID, computed on the whole recording
            if payload is not None:
                pf = payload[start:end][mask]
                X[li, 12] = pf[:, 0].max()
                X[li, 13] = pf[:, 1].max()
                X[li, 14] = pf[:, 2].mean()

            # ground truth: share of attack frames of this ID in the window
            node_atk[li] = w_atk[mask].mean()

        # CAN ID indices for embedding
        id_indices = np.array([vocab.get(cid, 0) for cid in unique_ids], dtype=np.int64)

        # Adjacency (temporal co-occurrence)
        adj = np.zeros((N, N), dtype=np.float64)
        local_idx = np.array([{cid: i for i, cid in enumerate(unique_ids)}[c] for c in w_ids])
        for k in range(W - 1):
            li, lj = local_idx[k], local_idx[k + 1]
            if li != lj:
                adj[li, lj] += 1.0
                adj[lj, li] += 1.0
        row_sums = adj.sum(axis=1, keepdims=True)
        adj_norm = adj / np.maximum(row_sums, 1e-8)

        # Label
        atk_count = int(w_atk.sum())
        label = attack_type_id if atk_count > 0 else 0

        window = {
            "x": torch.tensor(X, dtype=torch.float32),
            "adj": torch.tensor(adj_norm, dtype=torch.float32),
            "can_id_indices": torch.tensor(id_indices, dtype=torch.long),
            "can_ids": torch.tensor(unique_ids, dtype=torch.long),
            "y": torch.tensor([label], dtype=torch.long),
            "label": label,
            "label_name": INDEX_TO_LABEL[label],
            "num_nodes": N,
            "attack_ratio": atk_count / W,
            "t_start": float(w_ts[0]),          # start time, for temporal splits
            "x_raw": X.astype(np.float32),
            # raw sequence, kept only until the window features are computed
            "seq_ids": w_ids.astype(np.int64).copy(),
            "seq_ts": w_ts.astype(np.float64).copy(),
            "seq_dlc": dlc_data[start:end].copy(),
            "node_atk": torch.tensor(node_atk, dtype=torch.float32),
        }

        if label == 0:
            windows_normal.append(window)
        else:
            windows_attack.append(window)

    # Subsample normal
    if max_normal > 0 and len(windows_normal) > max_normal:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(windows_normal), size=max_normal, replace=False)
        windows_normal = [windows_normal[i] for i in sorted(idx)]

    return windows_attack + windows_normal


# ============================================================
# Split
# ============================================================

def stratified_split(
    windows: List[Dict], ratios: Tuple[float, float, float], seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for i, w in enumerate(windows):
        by_label[w["label"]].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in sorted(by_label.items()):
        arr = np.array(indices)
        rng.shuffle(arr)
        n = len(arr)
        nt = max(1, int(round(n * ratios[0])))
        nv = max(1, int(round(n * ratios[1])))
        nte = max(1, n - nt - nv)
        while nt + nv + nte > n:
            nt = max(1, nt - 1)
        while nt + nv + nte < n:
            nt += 1
        train_idx.extend(arr[:nt].tolist())
        val_idx.extend(arr[nt:nt+nv].tolist())
        test_idx.extend(arr[nt+nv:].tolist())

    return ([windows[i] for i in train_idx],
            [windows[i] for i in val_idx],
            [windows[i] for i in test_idx])


# ============================================================
# Normalization
# ============================================================

def compute_norm_stats(windows):
    all_x = np.concatenate([w["x"].numpy() for w in windows], axis=0)
    mean, std = all_x.mean(0), all_x.std(0)
    std[std < 1e-8] = 1.0
    return {"mean": mean, "std": std}


def apply_norm(windows, stats):
    m = torch.tensor(stats["mean"], dtype=torch.float32)
    s = torch.tensor(stats["std"], dtype=torch.float32)
    for w in windows:
        w["x"] = (w["x"] - m) / s
    return windows


# ============================================================
# Behavior features (per-ID normal profile)
# ============================================================

def compute_id_profiles(windows, min_count):
    """Per-CAN-ID mean/std of raw features over NORMAL training windows."""
    normal = [w for w in windows if w["label"] == 0]
    all_x = np.concatenate([w["x_raw"] for w in normal], axis=0).astype(np.float64)
    g_mean, g_std = all_x.mean(0), all_x.std(0)
    g_std[g_std < 1e-8] = 1.0

    acc = defaultdict(lambda: [0, np.zeros(NUM_FEATURES), np.zeros(NUM_FEATURES)])
    for w in normal:
        for cid, row in zip(w["can_id_indices"].tolist(), w["x_raw"].astype(np.float64)):
            a = acc[cid]
            a[0] += 1
            a[1] += row
            a[2] += row * row
    profiles = {}
    for cid, (n, s1, s2) in acc.items():
        if n < min_count:
            continue
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
        # floor: an ID whose feature is constant in normal traffic would
        # otherwise turn any tiny change into an enormous z-score
        std = np.maximum(std, 0.1 * g_std)
        profiles[cid] = (mean, std)
    return profiles, (g_mean, g_std)


SURPRISE_CAP = 3.0   # p = 1e-3; IDs with >= ~2000 normal values all reach the same ceiling


def _surprise(sorted_vals, v, signed):
    """Conformal two-sided tail surprise of values v w.r.t. a sorted reference sample."""
    n = len(sorted_vals)
    le = np.searchsorted(sorted_vals, v, side="right")          # #{ref <= v}
    ge = n - np.searchsorted(sorted_vals, v, side="left")       # #{ref >= v}
    p = np.minimum(1.0, 2.0 * (np.minimum(le, ge) + 1) / (n + 1))
    s = np.minimum(-np.log10(p), SURPRISE_CAP)     # same ceiling for every ID and vehicle
    if signed:
        s = np.sign(v - sorted_vals[n // 2]) * s
    return s


def compute_id_distributions(windows, min_count):
    """Per-CAN-ID sorted normal values of every raw feature (training split, normal only),
    plus the pooled distribution used for IDs without enough normal data."""
    normal = [w for w in windows if w["label"] == 0]
    rows = defaultdict(list)
    for w in normal:
        for cid, row in zip(w["can_id_indices"].tolist(), w["x_raw"].astype(np.float64)):
            rows[cid].append(row)
    dists = {cid: np.sort(np.array(r), axis=0) for cid, r in rows.items() if len(r) >= min_count}
    pooled = np.sort(np.concatenate([w["x_raw"] for w in normal], axis=0).astype(np.float64), axis=0)
    return dists, pooled


def apply_behavior_surprise(windows, dists, pooled, signed):
    """xb = conformal surprise of each node feature w.r.t. its ID's normal values, + unknown flag.
    Vectorised per CAN ID over all nodes of all windows."""
    sizes = [len(w["x_raw"]) for w in windows]
    X = np.concatenate([w["x_raw"] for w in windows], axis=0).astype(np.float64)
    cids = np.concatenate([w["can_id_indices"].numpy() for w in windows])
    S = np.zeros_like(X)
    unk = np.zeros((len(X), 1))
    for cid in np.unique(cids):
        m = cids == cid
        ref = dists.get(int(cid))
        if ref is None:
            ref = pooled
            unk[m, 0] = 1.0
        for k in range(X.shape[1]):
            S[m, k] = _surprise(ref[:, k], X[m, k], signed)
    out = np.concatenate([S, unk], axis=1)
    start = 0
    for w, n in zip(windows, sizes):
        w["xb"] = torch.tensor(out[start:start + n], dtype=torch.float32)
        del w["x_raw"]
        start += n
    return windows


def apply_behavior(windows, profiles, global_stats, clip=20.0):
    g_mean, g_std = global_stats
    for w in windows:
        xr = w["x_raw"].astype(np.float64)
        z = np.zeros_like(xr)
        unk = np.zeros((len(xr), 1))
        for i, cid in enumerate(w["can_id_indices"].tolist()):
            if cid in profiles:
                m, s = profiles[cid]
            else:
                m, s = g_mean, g_std
                unk[i, 0] = 1.0
            z[i] = (xr[i] - m) / s
        z = np.clip(z, -clip, clip)
        w["xb"] = torch.tensor(np.concatenate([z, unk], axis=1), dtype=torch.float32)
        del w["x_raw"]
    return windows


# ============================================================
# Window-level behavior features
# ============================================================

def build_window_reference(windows):
    """Normal-traffic reference from NORMAL training windows (raw IDs)."""
    normal = [w for w in windows if w["label"] == 0]
    known = np.zeros(NID, dtype=bool)
    count_sum, count_n = np.zeros(NID), np.zeros(NID)
    trans, gap_ids, gaps, dlc_ids, dlcs = [], [], [], [], []
    for w in normal:
        ids = np.clip(w["seq_ids"], 0, NID - 1)
        known[ids] = True
        u, c = np.unique(ids, return_counts=True)
        count_sum[u] += c
        count_n[u] += 1
        trans.append(ids[:-1] * NID + ids[1:])
        order = np.lexsort((w["seq_ts"], ids))
        si, st = ids[order], w["seq_ts"][order]
        same = si[1:] == si[:-1]
        gap_ids.append(si[1:][same])
        gaps.append(np.diff(st)[same])
        dlc_ids.append(ids)
        dlcs.append(w["seq_dlc"].astype(np.int64))
    exp_count = np.where(count_n > 0, count_sum / np.maximum(count_n, 1), 0.0)

    period = np.zeros(NID)
    g = pd.DataFrame({"id": np.concatenate(gap_ids), "gap": np.concatenate(gaps)})
    g = g[g["gap"] > 0]
    med = g.groupby("id")["gap"].median()
    period[med.index.values] = med.values

    dlc_mode = np.full(NID, -1, dtype=np.int64)
    d = pd.DataFrame({"id": np.concatenate(dlc_ids), "dlc": np.concatenate(dlcs)})
    mode = d.groupby("id")["dlc"].agg(lambda x: x.value_counts().idxmax())
    dlc_mode[mode.index.values] = mode.values

    return {"known": known, "exp_count": exp_count, "period": period,
            "dlc_mode": dlc_mode, "trans": np.unique(np.concatenate(trans)),
            "n_normal_windows": len(normal)}


def window_raw_features(w, ref):
    ids = np.clip(w["seq_ids"], 0, NID - 1)
    raw_ids = w["seq_ids"]
    ts, dlc = w["seq_ts"], w["seq_dlc"].astype(np.int64)
    n = len(ids)
    f = np.zeros(len(WINDOW_FEATURES))

    # +1 ID increments. Injected frames are interleaved with legitimate traffic,
    # so the successor (ID + 1) is searched within the next INC_LAG frames, and
    # chains of increments are followed across that gap.
    has_succ = np.zeros(n, dtype=bool)
    for lag in range(1, INC_LAG + 1):
        if lag < n:
            has_succ[:-lag] |= raw_ids[lag:] == raw_ids[:-lag] + 1
    f[0] = has_succ.mean() if n > 1 else 0.0
    chain = np.ones(n, dtype=np.int64)
    last_pos = {}
    for i, v in enumerate(raw_ids.tolist()):
        j = last_pos.get(v - 1)
        if j is not None and i - j <= INC_LAG:
            chain[i] = chain[j] + 1
        last_pos[v] = i
    f[1] = (chain.max() - 1) / max(n - 1, 1)

    known = ref["known"][ids] & (raw_ids < NID)
    f[2] = 1.0 - known.mean()
    codes = ids[:-1] * NID + ids[1:]
    f[3] = (~np.isin(codes, ref["trans"], assume_unique=False)).mean() if n > 1 else 0.0

    u, c = np.unique(ids, return_counts=True)
    ku = ref["known"][u] & (ref["exp_count"][u] > 0)
    if ku.any():
        ratio = c[ku] / ref["exp_count"][u[ku]]
        f[4] = np.log(ratio.max())
        f[5] = float(((ratio > 1.5) & (c[ku] >= 2)).sum())

    order = np.lexsort((ts, ids))
    si, st = ids[order], ts[order]
    same = si[1:] == si[:-1]
    gid, gap = si[1:][same], np.diff(st)[same]
    per = ref["period"][gid]
    ok = (per > 0) & ref["known"][gid]
    if ok.any():
        m = (gap[ok] / per[ok]).min()
        f[6] = -np.log10(max(m, 1e-4))

    if known.any():
        mode = ref["dlc_mode"][ids[known]]
        f[7] = ((dlc[known] != mode) & (mode >= 0)).mean()
    f[8] = len(u)
    return f


def apply_window_features(parts, ref, clip=20.0, norm="zscore", ref_windows=None):
    """ref_windows: the normal windows that define 'normal' (default: normal of parts[0])."""
    train = parts[0]
    raw = {id(w): window_raw_features(w, ref) for part in parts for w in part}
    if ref_windows is None:
        ref_windows = [w for w in train if w["label"] == 0]
    normal = np.stack([raw[id(w)] for w in ref_windows])
    mean, std = normal.mean(0), normal.std(0)
    std = np.maximum(std, 1e-2)       # several features are ~0 in normal traffic
    sorted_normal = np.sort(normal, axis=0)
    for part in parts:
        for w in part:
            r = raw[id(w)]
            if norm == "zscore":
                z = np.clip((r - mean) / std, -clip, clip)
            else:
                z = np.array([_surprise(sorted_normal[:, k], r[k:k + 1], norm == "surprise_signed")[0]
                              for k in range(len(r))])
            w["wb"] = torch.tensor(z, dtype=torch.float32)
            for k in ("seq_ids", "seq_ts", "seq_dlc"):
                del w[k]
    return {"mean": mean.tolist(), "std": std.tolist()}


def _auc(score, y):
    """Rank AUROC (Mann-Whitney), y in {0,1}."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


NODE_CLASSES = ["gear", "rpm", "speed", "standstill", "interval", "combined", "dos", "fuzzing"]
BASE_NAMES = ["freq", "iat_mean", "iat_std"] + [f"byte{i}_mean" for i in range(8)] + ["byte_change_rate"]


def node_separability_report(windows):
    """AUROC of each behavior feature (per-ID z-score) for attacked nodes of a class vs
    nodes of the SAME CAN IDs in normal windows: shows whether a node-level feature
    exposes the attack without knowing which vehicle it comes from."""
    names = BASE_NAMES + PAYLOAD_FEATURES[: NUM_FEATURES - 12]
    normal_rows = defaultdict(list)
    for w in windows:
        if w["label"] == 0:
            for cid, row in zip(w["can_id_indices"].tolist(), w["xb"][:, :NUM_FEATURES].numpy()):
                normal_rows[cid].append(row)
    out = {}
    print(f"  {'class':<12}{'n_atk':>7}{'best base feature':>34}" +
          "".join(f"{n:>24}" for n in PAYLOAD_FEATURES[: NUM_FEATURES - 12]))
    for cls in NODE_CLASSES:
        pos, cids = [], set()
        for w in windows:
            if w["label_name"] != cls:
                continue
            atk = w["node_atk"].numpy() > 0
            for cid, row, a in zip(w["can_id_indices"].tolist(), w["xb"][:, :NUM_FEATURES].numpy(), atk):
                if a:
                    pos.append(row)
                    cids.add(cid)
        neg = [r for c in cids for r in normal_rows.get(c, [])]
        if len(pos) < 5 or len(neg) < 5:
            continue
        X = np.vstack([np.array(pos), np.array(neg)])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)
        aucs = {}
        for k, nm in enumerate(names):
            a = _auc(X[:, k], y)
            aucs[nm] = round(max(a, 1 - a), 4)
        base_best = max(BASE_NAMES, key=lambda n: aucs[n])
        out[cls] = {"n_attacked_nodes": len(pos), "auroc": aucs}
        print(f"  {cls:<12}{len(pos):>7}{base_best + f' ({aucs[base_best]:.3f})':>34}" +
              "".join(f"{aucs[n]:>24.3f}" for n in names[12:]))
    return out


HARD_PAIRS = [("fuzzing", "systematic"), ("rpm", "standstill"), ("gear", "combined"),
              ("rpm", "combined"), ("speed", "combined"), ("normal", "interval"),
              ("normal", "speed"), ("normal", "dos"), ("normal", "gear"),
              ("normal", "combined"), ("normal", "ATTACK")]


def separability_report(train, val):
    """AUROC of the window features alone, per hard pair (fit on train, score on val)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        LogisticRegression = None

    def select(ws, a, b):
        if b == "ATTACK":
            sel = [w for w in ws if w["label_name"] == a or w["label"] != 0]
            y = np.array([w["label"] != 0 for w in sel], dtype=int)
        else:
            sel = [w for w in ws if w["label_name"] in (a, b)]
            y = np.array([w["label_name"] == b for w in sel], dtype=int)
        X = np.stack([w["wb"].numpy() for w in sel]) if sel else np.zeros((0, len(WINDOW_FEATURES)))
        return X, y

    rows = []
    print(f"\n  {'pair':<24s} {'n_val':>7s} {'LR AUROC':>9s}  best single feature")
    for a, b in HARD_PAIRS:
        Xtr, ytr = select(train, a, b)
        Xva, yva = select(val, a, b)
        if len(np.unique(yva)) < 2 or len(np.unique(ytr)) < 2:
            continue
        lr_auc = float("nan")
        if LogisticRegression is not None:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
            lr_auc = _auc(clf.decision_function(Xva), yva)
        singles = []
        for k, name in enumerate(WINDOW_FEATURES):
            a_k = _auc(Xva[:, k], yva)
            singles.append((max(a_k, 1 - a_k), name))
        best = max(singles)
        rows.append({"pair": f"{a} vs {b}", "n_val": int(len(yva)), "lr_auroc": lr_auc,
                     "best_feature": best[1], "best_feature_auroc": best[0],
                     "single_auroc": {n: round(v, 4) for v, n in singles}})
        print(f"  {a + ' vs ' + b:<24s} {len(yva):>7d} {lr_auc:>9.4f}  {best[1]} ({best[0]:.4f})")

    # per-class mean z of each feature (val), to read what each feature captures
    names = sorted({w["label_name"] for w in val}, key=lambda n: ATTACK_MAP[n])
    table = {}
    print(f"\n  mean z-score per class (val):")
    print("  " + f"{'feature':<24s}" + "".join(f"{n[:9]:>10s}" for n in names))
    for k, fname in enumerate(WINDOW_FEATURES):
        vals = []
        for nm in names:
            v = np.mean([w["wb"][k].item() for w in val if w["label_name"] == nm])
            vals.append(round(float(v), 2))
        table[fname] = dict(zip(names, vals))
        print("  " + f"{fname:<24s}" + "".join(f"{v:>10.2f}" for v in vals))
    return {"pairs": rows, "class_means": table}


# ============================================================
# Main pipeline
# ============================================================

def run(input_dir: Path, output_dir: Path, window_size: int,
        max_normal: int, seed: int, x_norm_source: str = "all",
        profile_min_count: int = 20, payload_features: bool = False,
        behavior_norm: str = "zscore", profile_files=None):
    global NUM_FEATURES
    NUM_FEATURES = 15 if payload_features else 12
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nPreprocessing: {input_dir}\n{'='*60}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {input_dir}")

    # First pass: build vocab from all files
    print("\n[Step 1] Building CAN ID vocabulary...")
    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    vocab = build_vocab(combined)
    print(f"  Vocabulary: {len(vocab)} CAN IDs")
    del combined

    # Second pass: build windows per file
    print(f"\n[Step 2] Building windows (size={window_size}, no overlap)...")
    all_windows = []
    file_stats = {}

    for fpath, df in zip(csv_files, all_dfs):
        stem = fpath.stem.lower()
        atk_id = 0
        for key, val in ATTACK_MAP.items():
            if key == stem or key in stem:
                atk_id = val
                break

        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        print(f"\n  [{stem}] {len(df):,} msgs (attack_class={atk_id})")

        byte_data = parse_payloads(df["data_field"])
        dlc_data = parse_dlc(df["data_field"])
        payload = (payload_stream_features(df["arbitration_id"].values.astype(np.int64), byte_data)
                   if payload_features else None)
        windows = compute_windows(df, byte_data, dlc_data, atk_id, vocab,
                                   window_size, 2, max_normal, seed, payload)

        n_atk = sum(1 for w in windows if w["label"] != 0)
        n_nrm = len(windows) - n_atk
        print(f"    → {len(windows)} windows (attack={n_atk}, normal={n_nrm})")
        file_stats[stem] = {"messages": len(df), "attack_class": atk_id,
                            "windows_attack": n_atk, "windows_normal": n_nrm}

        for w in windows:
            w["source_file"] = stem
        all_windows.extend(windows)

    del all_dfs

    print(f"\n[Total] {len(all_windows)} windows")
    label_dist = defaultdict(int)
    for w in all_windows:
        label_dist[w["label_name"]] += 1
    for lbl, cnt in sorted(label_dist.items()):
        print(f"  {lbl:>12s}: {cnt}")

    # Split
    print(f"\n[Step 3] Splitting...")
    train, val, test = stratified_split(all_windows, (0.6, 0.2, 0.2), seed)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Reference of NORMAL traffic: normal training windows, optionally only of the
    # recordings available at phase 1 (--profile_files), so that no phase sees a
    # normal reference built from recordings that arrive later (CL protocol)
    ref = [w for w in train if w["label"] == 0
           and (not profile_files or w.get("source_file") in profile_files)]
    print(f"\n[Reference] normal reference: {len(ref)} training windows from "
          f"{sorted({w.get('source_file') for w in ref})}")

    # Behavior features from per-ID normal profiles (training split, normal only)
    print(f"\n[Step 4a] Behavior features (per-ID normal profiles)...")
    profiles, g_stats = compute_id_profiles(ref, profile_min_count)
    if behavior_norm == "zscore":
        for part in (train, val, test):
            apply_behavior(part, profiles, g_stats)
    else:
        print(f"  behavior features as conformal surprise ({behavior_norm})")
        dists, pooled = compute_id_distributions(ref, profile_min_count)
        for part in (train, val, test):
            apply_behavior_surprise(part, dists, pooled, behavior_norm == "surprise_signed")
    n_unk = sum(int(w["xb"][:, -1].sum()) for w in train)
    n_nodes = sum(w["num_nodes"] for w in train)
    print(f"  Profiles for {len(profiles)} CAN IDs; unknown-ID nodes in train: "
          f"{n_unk}/{n_nodes} ({100 * n_unk / max(n_nodes, 1):.2f}%)")

    if payload_features:
        print("  Node-level separability of the behavior features (val, attacked nodes vs "
              "normal nodes of the same CAN IDs):")
        node_rep = node_separability_report(val)
        with open(output_dir / "node_report.json", "w") as f:
            json.dump(node_rep, f, indent=2)

    # Window-level behavior features (reference from normal training windows)
    print(f"\n[Step 4b] Window features...")
    wref = build_window_reference(ref)
    print(f"  Normal reference: {wref['n_normal_windows']} windows, "
          f"{int(wref['known'].sum())} known IDs, {len(wref['trans'])} known ID transitions")
    window_norm = apply_window_features((train, val, test), wref, norm=behavior_norm,
                                        ref_windows=ref)
    print("  Separability of the window features alone (val):")
    report = separability_report(train, val)
    with open(output_dir / "window_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Normalize
    print(f"\n[Step 4] Normalizing x (stats from: {x_norm_source})...")
    if profile_files:     # x statistics from the same normal reference (no later recordings)
        stats = compute_norm_stats(ref)
    else:
        stats = compute_norm_stats(train if x_norm_source == "all"
                                   else [w for w in train if w["label"] == 0])
    train = apply_norm(train, stats)
    val = apply_norm(val, stats)
    test = apply_norm(test, stats)

    # Save
    print(f"\n[Step 5] Saving...")
    for name, data in [("train", train), ("val", val), ("test", test)]:
        p = output_dir / f"windows_{name}.pt"
        torch.save(data, p)
        print(f"  {p} ({len(data)} windows)")

    metadata = {
        "window_size": window_size,
        "num_features": NUM_FEATURES,
        "num_classes": NUM_CLASSES,
        "vocab_size": len(vocab) + 1,  # +1 for UNK at index 0
        "vocab": {str(k): v for k, v in vocab.items()},
        "label_map": ATTACK_MAP,
        "norm_stats": {"mean": stats["mean"].tolist(), "std": stats["std"].tolist()},
        "file_stats": file_stats,
        "behavior_dim": NUM_FEATURES + 1,
        "payload_features": payload_features,
        "behavior_profiles": len(profiles),
        "profile_min_count": profile_min_count,
        "x_norm_source": x_norm_source,
        "behavior_norm": behavior_norm,
        "profile_files": sorted(profile_files) if profile_files else "all",
        "max_normal_per_file": max_normal,
        "window_dim": len(WINDOW_FEATURES),
        "window_features": WINDOW_FEATURES,
        "window_norm": window_norm,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json")

    print(f"\n{'='*60}\nDone!\n{'='*60}")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser("MoE-IVN-IDS Preprocessing (behavior + window features)")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--max_normal_per_file", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--x_norm_source", choices=["all", "normal"], default="all",
                   help="all: identical to preprocess.py (keeps old checkpoints valid); "
                        "normal: stats from normal training windows only (no attack leakage)")
    p.add_argument("--profile_min_count", type=int, default=20,
                   help="minimum normal occurrences for a CAN ID to get its own profile")
    p.add_argument("--profile_files", default="",
                   help="comma-separated recordings whose NORMAL training windows define the "
                        "normal reference (per-ID profiles, window reference, x statistics), e.g. "
                        "dos,fuzzing,gear,interval,rpm for the phase-1 recordings; empty = all")
    p.add_argument("--behavior_norm", choices=["zscore", "surprise", "surprise_signed"],
                   default="zscore",
                   help="behavior features relative to the vehicle's normal traffic: z-score "
                        "(previous runs) or conformal tail surprise (vehicle-invariant scale)")
    p.add_argument("--payload_features", action="store_true",
                   help="add 3 node features on payload continuity (jump, spike, bytes changed): "
                        "x has 15 features, xb 16")
    args = p.parse_args()

    run(Path(args.input_dir), Path(args.output_dir),
        args.window_size, args.max_normal_per_file, args.seed,
        args.x_norm_source, args.profile_min_count, args.payload_features, args.behavior_norm,
        {f.strip() for f in args.profile_files.split(",") if f.strip()})


if __name__ == "__main__":
    main()
