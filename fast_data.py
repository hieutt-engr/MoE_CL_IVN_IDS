"""
fast_data.py: GPU-resident graph batches, replacing DataLoader + collate_fn.

Why: each window is a small graph. The old pipeline rebuilt every batch on the
CPU (collate_fn, 4 workers) and then copied 3 x batch_size tiny tensors to the
GPU one by one. For a ~200K-parameter model that overhead dominates.

How: every split (train/val/test) is packed ONCE into zero-padded tensors

    X   [G, Nmax, F]     node features
    ID  [G, Nmax]        CAN-ID indices (int32, 0 = padding)
    A   [G, Nmax, Nmax]  adjacency (float16 by default, see below)
    n   [G]              number of real nodes per graph
    y   [G]              labels

using the existing collate_fn, so the window format does not need to be known.
The packed tensors are cached next to the data (packed_<split>.pt) and kept on
the GPU when they fit, otherwise in pinned CPU memory. A batch is then a few
index operations, trimmed to the largest graph in the batch.

Adjacency is stored as float16: under AMP, torch.mm/bmm cast it to float16
anyway, so this is lossless for training and halves the memory.

Subsets used by the training script (filtered phases, replay samples) are
still plain lists of window dicts; they are mapped back to rows of the packed
split by object identity, so no training logic has to change.
"""

import os
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from utils import WindowDataset, collate_fn


class PackedSplit:
    def __init__(self, windows, cache_path, device, src_path=None, adj_dtype=torch.float16,
                 placement="auto", gpu_fraction=0.5, num_workers=4, log=print):
        self.windows_index = {id(w): i for i, w in enumerate(windows)}
        cache_path = Path(cache_path)
        src_mtime = os.path.getmtime(src_path) if src_path else None
        tensors = self._load_or_build(windows, cache_path, src_mtime, adj_dtype, num_workers, log)
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
        self.X, self.ID, self.A = tensors["X"], tensors["ID"], tensors["A"]
        self.n, self.y = tensors["n"], tensors["y"]
        self.on_gpu, self.device = on_gpu, device
        log(f"  [FastData] {cache_path.name}: "
            f"{self.n_graphs} graphs, Nmax={self.X.size(1)}, "
            f"{nbytes / 2**20:.0f} MB on {'GPU' if on_gpu else 'pinned CPU'}")

    @staticmethod
    def _load_or_build(windows, cache_path, src_mtime, adj_dtype, num_workers, log):
        if cache_path.exists():
            t = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (t.get("n_graphs") == len(windows) and t["A"].dtype == adj_dtype
                    and t.get("src_mtime") == src_mtime):
                return {k: t[k] for k in ("X", "ID", "A", "n", "y")}
            log(f"  [FastData] cache {cache_path.name} is stale, rebuilding")

        log(f"  [FastData] packing {len(windows)} graphs (one-time) ...")
        # Collate in the main process. With DataLoader workers every tiny tensor
        # comes back through a shared-memory segment, and keeping ~100K of them
        # alive exhausts the mmap/file-descriptor limit ("unable to mmap ...").
        ds = WindowDataset(windows)
        xs, ids, adjs, ys = [], [], [], []
        step = 2048
        for start in range(0, len(ds), step):
            b = collate_fn([ds[i] for i in range(start, min(start + step, len(ds)))])
            if (start // step) % 10 == 0:
                log(f"  [FastData]   {start}/{len(ds)}")
            xs.extend(b["x_list"])
            ids.extend(b["id_idx_list"])
            adjs.extend(a.to_dense() if a.is_sparse else a for a in b["adj_list"])
            ys.append(b["y"])
        n = torch.tensor([x.size(0) for x in xs], dtype=torch.int32)
        N = int(n.max())
        X = pad_sequence(xs, batch_first=True).float()
        ID = pad_sequence([i.to(torch.int32) for i in ids], batch_first=True)
        A = torch.zeros(len(adjs), N, N, dtype=adj_dtype)
        for g, a in enumerate(adjs):
            k = a.size(0)
            A[g, :k, :k] = a.to(adj_dtype)
        t = {"X": X, "ID": ID, "A": A, "n": n, "y": torch.cat(ys).long(),
             "n_graphs": len(windows), "src_mtime": src_mtime}
        tmp = cache_path.with_suffix(".tmp")
        torch.save(t, tmp)
        os.replace(tmp, cache_path)
        return {k: t[k] for k in ("X", "ID", "A", "n", "y")}

    def rows_for(self, windows):
        try:
            return [self.windows_index[id(w)] for w in windows]
        except KeyError:
            return None


class GPUBatches:
    """Drop-in replacement for DataLoader(WindowDataset(windows), ...).

    Yields dicts with padded tensors already on the device:
      x [B,Nb,F], ids [B,Nb], adj [B,Nb,Nb], n [B], y [B]
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
