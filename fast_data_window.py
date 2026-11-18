"""
fast_data.py: GPU-resident graph batches, replacing DataLoader + collate_fn.

Every split (train/val/test) is packed ONCE into zero-padded tensors

    X   [G, Nmax, F]     node features ("x")
    ID  [G, Nmax]        CAN-ID indices (int32, 0 = padding)
    A   [G, Nmax, Nmax]  adjacency (float16: under AMP mm/bmm use float16 anyway)
    n   [G]              number of real nodes per graph
    y   [G]              labels
  optional, when the windows carry them (preprocess_behavior.py):
    XB  [G, Nmax, Fb]    behavior features ("xb")
    NA  [G, Nmax]        per-node share of attack frames ("node_atk", float16)
  optional, when the windows carry it (preprocess_window.py):
    WB  [G, Fw]          window-level behavior features ("wb")

directly from the window dicts written by preprocess.py, in the main process
(no DataLoader workers, so no shared-memory limits). The packed tensors are
cached next to the data (packed_<split>.pt), kept on the GPU when they fit,
otherwise in pinned CPU memory. A batch is a few index operations, trimmed to
the largest graph in the batch.

Subsets used by the training script (filtered phases, replay samples) are
plain lists of window dicts; they are mapped back to rows of the packed split
by object identity.
"""

import os
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

REQUIRED = ("X", "ID", "A", "n", "y")
OPTIONAL = ("XB", "NA", "WB")


class PackedSplit:
    def __init__(self, windows, cache_path, device, src_path=None, adj_dtype=torch.float16,
                 placement="auto", gpu_fraction=0.5, num_workers=0, log=print):
        self.windows_index = {id(w): i for i, w in enumerate(windows)}
        cache_path = Path(cache_path)
        src_mtime = os.path.getmtime(src_path) if src_path else None
        tensors = self._load_or_build(windows, cache_path, src_mtime, adj_dtype, log)
        self.n_graphs = tensors["y"].numel()

        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        on_gpu = False
        if device.type == "cuda" and placement in ("auto", "gpu"):
            free, _ = torch.cuda.mem_get_info(device)
            on_gpu = placement == "gpu" or nbytes < gpu_fraction * free
        if on_gpu:
            tensors = {k: v.to(device) for k, v in tensors.items()}
        elif device.type == "cuda":
            tensors = {k: v.pin_memory() for k, v in tensors.items()}
        self.t = tensors
        self.X, self.ID, self.A = tensors["X"], tensors["ID"], tensors["A"]
        self.n, self.y = tensors["n"], tensors["y"]
        self.XB, self.NA, self.WB = tensors.get("XB"), tensors.get("NA"), tensors.get("WB")
        self.on_gpu, self.device = on_gpu, device
        extra = (" +behavior" if self.XB is not None else "") + (" +window" if self.WB is not None else "")
        log(f"  [FastData] {cache_path.name}: {self.n_graphs} graphs, Nmax={self.X.size(1)}{extra}, "
            f"{nbytes / 2**20:.0f} MB on {'GPU' if on_gpu else 'pinned CPU'}")

    @staticmethod
    def _load_or_build(windows, cache_path, src_mtime, adj_dtype, log):
        has_xb = bool(windows) and "xb" in windows[0]
        has_wb = bool(windows) and "wb" in windows[0]
        if cache_path.exists():
            t = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (t.get("n_graphs") == len(windows) and t["A"].dtype == adj_dtype
                    and t.get("src_mtime") == src_mtime and ("XB" in t) == has_xb
                    and ("WB" in t) == has_wb):
                return {k: t[k] for k in REQUIRED + OPTIONAL if k in t}
            log(f"  [FastData] cache {cache_path.name} is stale, rebuilding")

        log(f"  [FastData] packing {len(windows)} graphs (one-time) ...")
        xs, ids, xbs, nas = [], [], [], []
        n = torch.empty(len(windows), dtype=torch.int32)
        for g, w in enumerate(windows):
            xs.append(w["x"].float())
            ids.append(w["can_id_indices"].to(torch.int32))
            n[g] = w["x"].size(0)
            if has_xb:
                xbs.append(w["xb"].float())
                nas.append(w["node_atk"].to(torch.float16))
        N = int(n.max())
        A = torch.zeros(len(windows), N, N, dtype=adj_dtype)
        for g, w in enumerate(windows):
            a = w["adj"]
            a = a.to_dense() if a.is_sparse else a
            k = a.size(0)
            A[g, :k, :k] = a.to(adj_dtype)
            if g % 20000 == 0:
                log(f"  [FastData]   {g}/{len(windows)}")
        t = {"X": pad_sequence(xs, batch_first=True), "ID": pad_sequence(ids, batch_first=True),
             "A": A, "n": n, "y": torch.tensor([w["label"] for w in windows], dtype=torch.long),
             "n_graphs": len(windows), "src_mtime": src_mtime}
        if has_wb:
            t["WB"] = torch.stack([w["wb"].float() for w in windows])
        if has_xb:
            t["XB"] = pad_sequence(xbs, batch_first=True)
            t["NA"] = pad_sequence(nas, batch_first=True)
        tmp = cache_path.with_suffix(".tmp")
        torch.save(t, tmp)
        os.replace(tmp, cache_path)
        return {k: t[k] for k in REQUIRED + OPTIONAL if k in t}

    def rows_for(self, windows):
        try:
            return [self.windows_index[id(w)] for w in windows]
        except KeyError:
            return None


class GPUBatches:
    """Drop-in replacement for DataLoader(WindowDataset(windows), ...).

    Yields dicts with padded tensors on the device:
      x [B,Nb,F], ids [B,Nb], adj [B,Nb,Nb], n [B], y [B]
      (+ xb [B,Nb,Fb], node_atk [B,Nb] when packed)
    where Nb is the largest graph in the batch.
    """

    def __init__(self, pack, rows, batch_size, shuffle=False, drop_last=False):
        self.pack = pack
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last

    def __len__(self):
        n = len(self.rows)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        p = self.pack
        rows = self.rows[torch.randperm(len(self.rows))] if self.shuffle else self.rows
        if p.on_gpu:
            rows = rows.to(p.device)
        for k in range(len(self)):
            r = rows[k * self.batch_size:(k + 1) * self.batch_size]
            n = p.n[r]
            Nb = int(n.max())                       # one small sync per batch
            batch = {"x": p.X[r, :Nb], "ids": p.ID[r, :Nb],
                     "adj": p.A[r, :Nb, :Nb], "n": n.long(), "y": p.y[r]}
            if p.XB is not None:
                batch["xb"] = p.XB[r, :Nb]
                batch["node_atk"] = p.NA[r, :Nb]
            if p.WB is not None:
                batch["wb"] = p.WB[r]
            if not p.on_gpu:
                batch = {k2: v.pin_memory().to(p.device, non_blocking=True)
                         for k2, v in batch.items()}
            yield batch


def lists_from_padded(batch):
    """Convert a padded batch back to (x_list, adj_list, id_idx_list) for models
    that only implement the list-based forward. Views, no copies."""
    ns = batch["n"].tolist()
    x, a, ids = batch["x"], batch["adj"], batch["ids"]
    return ([x[i, :k] for i, k in enumerate(ns)],
            [a[i, :k, :k].float() for i, k in enumerate(ns)],
            [ids[i, :k].long() for i, k in enumerate(ns)])
