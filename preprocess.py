#!/usr/bin/env python3
"""
preprocess.py — Lightweight preprocessing for MoE-IVN-IDS

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
        --output_dir ./data/2011-chevrolet-impala/preprocessed \
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
    attack_type_id: int,
    vocab: Dict[int, int],
    window_size: int,
    min_unique_ids: int,
    max_normal: int,
    seed: int,
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
# Main pipeline
# ============================================================

def run(input_dir: Path, output_dir: Path, window_size: int,
        max_normal: int, seed: int):
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
        windows = compute_windows(df, byte_data, atk_id, vocab,
                                   window_size, 2, max_normal, seed)

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

    # Normalize
    print(f"\n[Step 4] Normalizing...")
    stats = compute_norm_stats(train)
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
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json")

    print(f"\n{'='*60}\nDone!\n{'='*60}")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser("MoE-IVN-IDS Preprocessing")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--max_normal_per_file", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run(Path(args.input_dir), Path(args.output_dir),
        args.window_size, args.max_normal_per_file, args.seed)


if __name__ == "__main__":
    main()
