"""
utils.py — Shared utilities for MoE-IVN-IDS
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

ATTACK_MAP = {
    "normal": 0, "dos": 1, "fuzzing": 2, "gear": 3, "interval": 4,
    "rpm": 5, "speed": 6, "standstill": 7, "systematic": 8, "combined": 9,
}
INDEX_TO_LABEL = {v: k for k, v in ATTACK_MAP.items()}
NUM_CLASSES = len(ATTACK_MAP)


def get_class_names():
    return [INDEX_TO_LABEL[i] for i in range(NUM_CLASSES)]


# ============================================================
# Data
# ============================================================

def load_windows(path):
    data = torch.load(path, weights_only=False)
    print(f"  Loaded {len(data)} windows from {path}")
    return data


def load_metadata(path):
    with open(path) as f:
        return json.load(f)


class WindowDataset(torch.utils.data.Dataset):
    """
    Returns numpy arrays (not tensors) so DataLoader workers
    use pickle instead of shared memory — fixes mmap OOM.
    """
    def __init__(self, windows):
        # Convert all tensors to numpy at init time
        self.data = []
        for w in windows:
            self.data.append({
                "x": w["x"].numpy() if isinstance(w["x"], torch.Tensor) else w["x"],
                "adj": w["adj"].numpy() if isinstance(w["adj"], torch.Tensor) else w["adj"],
                "can_id_indices": w["can_id_indices"].numpy() if isinstance(w["can_id_indices"], torch.Tensor) else w["can_id_indices"],
                "y": int(w["y"].item()) if isinstance(w["y"], torch.Tensor) else int(w["y"]),
                "label": int(w["label"]),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Convert numpy arrays to tensors here (main process only)."""
    return {
        "x_list": [torch.from_numpy(w["x"]).float() for w in batch],
        "adj_list": [torch.from_numpy(w["adj"]).float() for w in batch],
        "id_idx_list": [torch.from_numpy(w["can_id_indices"]).long() for w in batch],
        "y": torch.tensor([w["y"] for w in batch], dtype=torch.long),
        "labels": [w["label"] for w in batch],
    }


def compute_class_weights(windows, num_classes, device):
    counts = torch.zeros(num_classes)
    for w in windows:
        lbl = w["y"].item() if isinstance(w["y"], torch.Tensor) else int(w["y"])
        counts[lbl] += 1
    raw = counts.sum() / (num_classes * (counts + 1e-6))
    return torch.sqrt(raw).to(device)


# ============================================================
# Metrics
# ============================================================

import warnings  # thêm ở đầu file nếu chưa có

def compute_metrics(y_true, y_pred):
    labels = sorted(set(y_true))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "acc": float(accuracy_score(y_true, y_pred)),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        }


def print_report(y_true, y_pred, logger=None):
    names = get_class_names()
    labels = sorted(set(y_true))
    target_names = [names[i] for i in labels]
    r = classification_report(y_true, y_pred, labels=labels,
                               target_names=target_names, digits=4, zero_division=0)
    if logger:
        logger.info(f"\n{r}")
    else:
        print(r)
    return r


# ============================================================
# Checkpoint
# ============================================================

def save_ckpt(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ============================================================
# Logger
# ============================================================

def setup_logger(save_dir, name="train"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(Path(save_dir) / f"{name}.log")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)