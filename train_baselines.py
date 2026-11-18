#!/usr/bin/env python3
"""
train_baselines.py: continual-learning baselines under exactly the same setting as
train_continual_window.py (final model A).

Shared with the main trainer (imported, not copied): data loading and packing, phase
orders, normal protocol 'recording' (the normal windows of a new recording arrive with
it), the DoS variant split, the bounded replay memory (reservoir or class-balanced,
M windows), checkpoint selection on validation macro F1, metrics and the output files
(results.json, cl_summary.json, preds_phase*.npz, log). Only the network and the
continual-learning method change.

Network for every baseline (DenseGraphIDS): a dense graph encoder that receives the same
inputs as the two-branch MoE (node features x, CAN-ID embedding with unknown IDs mapped
to the shared padding embedding, behavior features xb, window features wb), two graph
layers [h, Ah, h - Ah], mean + max readout, the same classifier MLP and cosine head.
--hidden 184 gives about 280k parameters, close to the two-branch MoE after expansion.

Methods (--method):
  finetune  new data only (lower bound)
  joint     retrain from scratch on all data seen so far at every phase (upper bound)
  er        experience replay with the shared bounded memory
  cbrs      er with a class-balanced memory (CBRS-like, as in ECBRS)
  ewc       online EWC penalty (+ replay unless --no_replay)
  lwf       learning without forgetting, KD on old-class logits (+ replay unless --no_replay)
  icarl     replay + KD on old classes, nearest-mean-of-exemplars classifier at test time
  derpp     DER++: replay stream with stored logits, alpha * MSE(logits) + beta * CE
  der       DER (dynamically expandable representation): a new encoder per phase, old
            encoders frozen, classifier on the concatenated features + auxiliary loss

    python train_baselines.py --method er --data_dir ./data/2011-chevrolet-impala/preprocessed_v3_variant \
        --save_dir ./save/baselines/impala/er_variant_s42 --seed 42

Run from the project root (imports train_continual_window.py).
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_continual_window as T
from networks.moe_ivn_ids_cosine import CosineLinear

CLASS_NAMES, NC = T.CLASS_NAMES, T.NC
REPLAY_METHODS = {"er", "cbrs", "ewc", "lwf", "icarl", "der"}      # concatenated replay
KD_METHODS = {"lwf", "icarl"}


# ============================================================================ network
class GraphEncoder(nn.Module):
    """Dense graph encoder over the same inputs as the two-branch MoE."""

    def __init__(self, fx, fb, fw, vocab, embed_dim=16, hidden=184, layers=2, dropout=0.2,
                 mask_unknown_ids=True):
        super().__init__()
        self.mask_unknown_ids = mask_unknown_ids
        self.id_embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.inp = nn.Sequential(nn.Linear(fx + embed_dim + fb + fw, hidden), nn.SiLU(),
                                 nn.LayerNorm(hidden))
        self.mix = nn.ModuleList([nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(),
                                                nn.Dropout(dropout), nn.Linear(hidden, hidden))
                                  for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out_dim = 2 * hidden

    def forward(self, x, ids, adj, n, xb, wb):
        B, N, _ = x.shape
        m = (torch.arange(N, device=x.device)[None, :] < n[:, None]).unsqueeze(-1)
        ids = ids.long()
        if self.mask_unknown_ids and xb is not None:
            ids = ids.masked_fill(xb[..., -1] > 0.5, 0)
        parts = [x.float(), self.id_embedding(ids).float(), xb.float(),
                 wb.float()[:, None, :].expand(B, N, -1)]
        h = self.inp(torch.cat(parts, dim=-1)) * m
        for lin, norm in zip(self.mix, self.norms):
            ah = torch.bmm(adj.to(h.dtype), h)
            h = norm(h + lin(torch.cat([h, ah, h - ah], dim=-1))) * m
        mean = h.sum(1) / n[:, None].to(h.dtype)
        mx = h.masked_fill(~m, float("-inf")).amax(1)
        return torch.cat([mean, mx], dim=-1)


class DenseGraphIDS(nn.Module):
    """Encoder(s) + LayerNorm + classifier MLP + cosine head. With several encoders (DER)
    the readouts are concatenated; only the last encoder is trainable."""
    needs_behavior = True

    def __init__(self, meta, args):
        super().__init__()
        self.window_dim = meta["window_dim"]
        self.enc_kw = dict(fx=meta["num_features"], fb=meta.get("behavior_dim", 13), fw=meta["window_dim"],
                           vocab=meta["vocab_size"], embed_dim=args.embed_dim, hidden=args.hidden,
                           layers=args.layers, dropout=args.dropout, mask_unknown_ids=not args.no_mask_unknown_ids)
        self.cls_hidden, self.dropout = args.cls_hidden, args.dropout
        self.encoders = nn.ModuleList([GraphEncoder(**self.enc_kw)])
        self._build_head()
        self.aux_head = None              # DER auxiliary classifier (new classes + "old")
        self.nme_means = None             # iCaRL: class means of normalised features
        self.use_nme = False

    def _build_head(self):
        d = sum(e.out_dim for e in self.encoders)
        self.readout_norm = nn.LayerNorm(d)
        self.classifier = nn.Sequential(nn.Linear(d, self.cls_hidden), nn.SiLU(), nn.Dropout(self.dropout),
                                        nn.Linear(self.cls_hidden, self.cls_hidden // 2))
        self.cos_head = CosineLinear(self.cls_hidden // 2, NC)

    def add_encoder(self, device):
        """DER: new trainable encoder (copy of the last one), old encoders frozen, new head."""
        new = copy.deepcopy(self.encoders[-1]).to(device)
        for e in self.encoders:
            for p in e.parameters():
                p.requires_grad = False
        self.encoders.append(new)
        self._build_head()
        self.readout_norm.to(device); self.classifier.to(device); self.cos_head.to(device)
        self.aux_head = nn.Sequential(nn.Linear(new.out_dim, self.cls_hidden // 2), nn.SiLU(),
                                      nn.Linear(self.cls_hidden // 2, NC + 1)).to(device)

    def train(self, mode=True):
        super().train(mode)
        for e in self.encoders[:-1]:          # frozen DER encoders: no dropout
            e.eval()
        return self

    def forward_padded(self, x, ids, adj, n, xb=None, wb=None, node_atk=None, **_):
        reads = [e(x, ids, adj, n, xb, wb) for e in self.encoders]
        z = self.readout_norm(torch.cat(reads, dim=-1))
        features = self.classifier(z)
        if self.use_nme and self.nme_means is not None:
            f = F.normalize(features.float(), dim=-1)
            logits = 16.0 * f @ self.nme_means.t()
            logits = logits.masked_fill(self.nme_means.abs().sum(1)[None, :] == 0, -1e4)
        else:
            logits = self.cos_head(features)
        probs = F.softmax(logits.float(), dim=-1)
        out = {"logits": logits, "features": features,
               "entropy": torch.zeros(x.size(0), device=x.device),
               "cls_entropy": -(probs * torch.log(probs + 1e-12)).sum(-1)}
        if self.aux_head is not None and self.training:
            out["aux_logits"] = self.aux_head(reads[-1])
        return out

    def get_info(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": train, "encoders": len(self.encoders)}


# ============================================================================ training
def kd_loss(new_logits, old_logits, old_idx, T_):
    if len(old_idx) == 0:
        return new_logits.new_zeros(())
    p_old = F.softmax(old_logits[:, old_idx].float() / T_, dim=-1)
    logp_new = F.log_softmax(new_logits[:, old_idx].float() / T_, dim=-1)
    return F.kl_div(logp_new, p_old, reduction="batchmean") * (T_ ** 2)


class EWC:
    """Online EWC: Fisher accumulated over phases, anchor = parameters after each phase."""

    def __init__(self):
        self.fisher, self.anchor = None, None

    def penalty(self, model):
        if self.fisher is None:
            return None
        loss = 0.0
        for n_, p in model.named_parameters():
            if p.requires_grad and n_ in self.fisher:
                loss = loss + (self.fisher[n_] * (p - self.anchor[n_]) ** 2).sum()
        return loss

    def update(self, model, windows, args, device, gamma=1.0):
        model.eval()
        fisher = {n_: torch.zeros_like(p) for n_, p in model.named_parameters() if p.requires_grad}
        loader = T.make_loader(windows, args, shuffle=True)
        nb = 0
        for batch in loader:
            if nb >= args.fisher_batches:
                break
            model.zero_grad(set_to_none=True)
            out = T.forward_batch(model, batch, device)
            F.cross_entropy(out["logits"].float(), batch["y"].to(device)).backward()
            for n_, p in model.named_parameters():
                if p.grad is not None and n_ in fisher:
                    fisher[n_] += p.grad.detach() ** 2
            nb += 1
        model.zero_grad(set_to_none=True)
        for n_ in fisher:
            fisher[n_] /= max(nb, 1)
            if self.fisher is not None and n_ in self.fisher:
                fisher[n_] += gamma * self.fisher[n_]
        self.fisher = fisher
        self.anchor = {n_: p.detach().clone() for n_, p in model.named_parameters() if p.requires_grad}


class ReplayStream:
    """DER++: endless shuffled batches from the memory, with the logits stored for them."""

    def __init__(self, windows, logits, args):
        self.windows, self.logits, self.args = windows, logits, args
        self.rng = np.random.default_rng(args.seed + 99)
        self._it = None

    def _new_epoch(self):
        perm = self.rng.permutation(len(self.windows))
        self.order = [self.windows[i] for i in perm]
        self.order_logits = self.logits[perm]
        self.k = 0
        self._it = iter(T.make_loader(self.order, self.args, shuffle=False))

    def next(self, device):
        if self._it is None:
            self._new_epoch()
        try:
            batch = next(self._it)
        except StopIteration:
            self._new_epoch()
            batch = next(self._it)
        bs = batch["y"].size(0)
        stored = self.order_logits[self.k:self.k + bs].to(device)
        exp_y = torch.tensor([w["label"] for w in self.order[self.k:self.k + bs]], device=device)
        if not torch.equal(exp_y, batch["y"].to(device).long()):
            raise RuntimeError("DER++ replay stream: batch order does not match the stored logits")
        self.k += bs
        return batch, stored


def fit(model, train_data, val_set, seen, phase_name, epochs, lr, args, device, logger, method,
        old_model=None, old_idx=(), ewc=None, stream=None, logit_adj=None, new_labels=()):
    """Training with the same checkpoint selection as the main trainer (best val macro F1)."""
    train_loader = T.make_loader(train_data, args, shuffle=True, drop_last=True)
    val_loader = T.make_loader(val_set, args, shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]
    if model.aux_head is not None:
        params += [p for p in model.aux_head.parameters() if p.requires_grad and all(p is not q for q in params)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-5)
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt = os.path.join(args.save_dir, f"{phase_name}_best.pth")
    old_idx_t = torch.tensor(sorted(old_idx), device=device, dtype=torch.long)
    new_t = torch.tensor(sorted(new_labels), device=device, dtype=torch.long)
    best, log_every = -1.0, max(1, epochs // 5)
    for ep in range(1, epochs + 1):
        model.train()
        tot = torch.zeros((), device=device)
        for batch in train_loader:
            y = batch["y"].to(device).long()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = T.forward_batch(model, batch, device)
                logits = out["logits"].float()
                loss = F.cross_entropy(logits + (logit_adj if logit_adj is not None else 0), y)
                if old_model is not None and method in KD_METHODS:
                    with torch.no_grad():
                        old_logits = T.forward_batch(old_model, batch, device)["logits"]
                    loss = loss + args.kd_weight * kd_loss(logits, old_logits, old_idx_t, args.kd_T)
                if "aux_logits" in out:                         # DER auxiliary loss
                    aux_y = torch.where(torch.isin(y, new_t), y, torch.full_like(y, NC))
                    loss = loss + args.der_aux_weight * F.cross_entropy(out["aux_logits"].float(), aux_y)
                if stream is not None:                          # DER++
                    rb, stored = stream.next(device)
                    r_out = T.forward_batch(model, rb, device)
                    r_logits = r_out["logits"].float()
                    loss = loss + args.derpp_alpha * F.mse_loss(r_logits, stored.float())
                    loss = loss + args.derpp_beta * F.cross_entropy(r_logits, rb["y"].to(device).long())
            if ewc is not None:
                pen = ewc.penalty(model)
                if pen is not None:
                    loss = loss + 0.5 * args.ewc_lambda * pen
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.detach()
        sch.step()
        if ep % args.eval_every == 0 or ep == epochs:
            m, _ = T.evaluate(model, val_loader, device, args, seen)
            if m["macro_f1"] > best:
                best = m["macro_f1"]
                T.save_ckpt({"model": model.state_dict(), "args": vars(args), "phase": phase_name,
                             "encoders": len(model.encoders)}, ckpt)
            if ep % log_every == 0 or ep == epochs:
                logger.info(f"  Ep {ep} | loss={tot.item() / max(len(train_loader), 1):.4f} "
                            f"val_F1={m['macro_f1']:.4f}")
    state = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state, strict=False)
    return best


@torch.no_grad()
def window_logits(model, windows, args, device):
    model.eval()
    outs = []
    for batch in T.make_loader(windows, args, shuffle=False):
        outs.append(T.forward_batch(model, batch, device)["logits"].float().cpu())
    return torch.cat(outs) if outs else torch.zeros(0, NC)


@torch.no_grad()
def class_means(model, windows, args, device):
    """iCaRL: mean of L2-normalised features per class, normalised again."""
    model.eval()
    feats, ys = [], []
    for batch in T.make_loader(windows, args, shuffle=False):
        out = T.forward_batch(model, batch, device)
        feats.append(F.normalize(out["features"].float(), dim=-1)); ys.append(batch["y"].to(device))
    f, y = torch.cat(feats), torch.cat(ys)
    means = torch.zeros(NC, f.size(1), device=device)
    for c in torch.unique(y):
        means[c] = F.normalize(f[y == c].mean(0), dim=0)
    return means


# ============================================================================ main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   choices=["finetune", "joint", "er", "cbrs", "ewc", "lwf", "icarl", "derpp", "der"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--phase_config", default="variant", choices=list(T.PHASE_CONFIGS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--fast_data", default="auto")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--save_preds", action=argparse.BooleanOptionalAction, default=True)
    # network
    p.add_argument("--hidden", type=int, default=156)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--cls_hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no_mask_unknown_ids", action="store_true",
                   help="give unknown CAN IDs their own embedding (the final model masks them)")
    # optimisation (same defaults as the final model)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--phase1_epochs", type=int, default=100)
    p.add_argument("--phase_epochs", type=int, default=60)
    p.add_argument("--phase_lr_mult", type=float, default=0.5)
    # data and replay (same as the final model)
    p.add_argument("--normal_protocol", default="recording", choices=["recording", "all"])
    p.add_argument("--memory_size", type=int, default=3000)
    p.add_argument("--memory_policy", default="reservoir",
                   choices=["reservoir", "class_balanced", "benign_anchored", "prior"])
    p.add_argument("--no_replay", action="store_true", help="ewc / lwf without replay (original form)")
    p.add_argument("--logit_adjust", type=float, default=0.0,
                   help="logit adjustment tau (the final model uses 0.6; 0 = plain baselines)")
    # method hyper-parameters
    p.add_argument("--ewc_lambda", type=float, default=100.0)
    p.add_argument("--fisher_batches", type=int, default=50)
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--kd_weight", type=float, default=1.0)
    p.add_argument("--derpp_alpha", type=float, default=0.5)
    p.add_argument("--derpp_beta", type=float, default=0.5)
    p.add_argument("--der_aux_weight", type=float, default=1.0)
    a = p.parse_args()
    # fields used by the shared BoundedMemory
    a.stratify_normal, a.anchor_share, a.min_per_class_frac = False, 0.5, 0.02
    if a.method == "cbrs":
        a.memory_policy = "class_balanced"
    return a


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    T.set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    T.acquire_lock(args.save_dir)
    args.run_name = args.run_name or Path(args.save_dir).resolve().name
    args.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    args.log_name = f"{args.run_name}_{args.method}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = T.setup_logger(args.save_dir, args.log_name)
    logger.info(f"Run: {args.run_name} | method {args.method} | phase config {args.phase_config} | {device}")
    logger.info(json.dumps(vars(args), indent=2))

    data = Path(args.data_dir)
    meta = T.load_metadata(data / "metadata.json")
    all_train, all_val, all_test = (T.load_windows(data / f"windows_{s}.pt") for s in ("train", "val", "test"))
    if args.fast_data != "off":
        for split, ws in (("train", all_train), ("val", all_val), ("test", all_test)):
            T._PACKS.append(T.PackedSplit(ws, data / f"packed_{split}.pt", device,
                                          src_path=data / f"windows_{split}.pt", placement=args.fast_data,
                                          num_workers=args.num_workers, log=logger.info))
    T.set_seed(args.seed)

    phases = T.PHASE_CONFIGS[args.phase_config]
    memory = T.BoundedMemory(args)
    uses_replay = (args.method in REPLAY_METHODS and not args.no_replay) or args.method == "derpp"
    results, seen, recordings, all_seen_train = [], set(), set(), []
    ewc = EWC() if args.method == "ewc" else None
    stored_logits = {}                     # DER++: id(window) -> logits when it entered the memory

    model = DenseGraphIDS(meta, args).to(device)
    logger.info(f"  Model: {model.get_info()}")

    def record(phase, test_windows, t0, decision):
        out = T.predict(model, T.make_loader(test_windows, args), device, args)
        m, per = T.metrics_from(out, seen)
        info = model.get_info()
        logger.info(f"\n  {phase['name']} result: F1={m['macro_f1']:.4f} acc={m['acc']:.4f} "
                    f"normal_FPR={m['normal_fpr']:.4f} detection={m['detection_rate']:.4f} "
                    f"params={info['total']:,} | {(time.time() - t0) / 60:.1f} min")
        logger.info(f"  Per-attack: {json.dumps({k: round(v, 4) for k, v in per.items()})}")
        hc = T.hard_confusions(out, seen)
        if hc["pairs"]:
            logger.info("  Hard confusions (test, true->pred): " + ", ".join(f"{k}={v}" for k, v in hc["pairs"].items()))
        logger.info(f"  Normal false alarms go to: {hc['normal_false_alarms']}")
        if args.save_preds:
            np.savez_compressed(os.path.join(args.save_dir, f"preds_{phase['name']}.npz"),
                                y=out["y"], logits=out["logits"].astype(np.float16))
        results.append({"phase": phase["name"], "desc": phase["desc"], "type": phase.get("type", "new"),
                        "mode": args.method, "overall": m, "per_attack": per, "num_experts": 0,
                        "params": info["total"], "encoders": info["encoders"], "decision_info": decision,
                        "specialization": None, "hard_confusions": hc, "minutes": (time.time() - t0) / 60})

    for pi, phase in enumerate(phases):
        t0 = time.time()
        logger.info(f"\n{'=' * 60}\n{phase['name']}: {phase['desc']}\n{'=' * 60}")
        seen_before = set(seen)
        new = set(phase["attacks"])
        if pi == 0:
            seen.update(new)
            recordings |= {CLASS_NAMES[c] for c in new if c != 0}
            train_new = T.select_windows(all_train, seen, recordings, args.normal_protocol)
            val_set = T.select_windows(all_val, seen, recordings, args.normal_protocol)
            test_set = T.select_windows(all_test, seen, recordings, args.normal_protocol)
            if phase.get("use_dos_part") == 1:
                train_new, val_set, test_set = (T.split_dos_windows(x, 1) for x in (train_new, val_set, test_set))
            ctx = []
        else:
            train_new = T.filter_by_labels(all_train, new)
            if phase.get("type") == "variant" and phase.get("use_dos_part") == 2:
                train_new = T.split_dos_windows(train_new, 2)
            seen.update(new)
            new_recs = {CLASS_NAMES[c] for c in new if c != 0} - recordings
            recordings |= new_recs
            val_set = T.select_windows(all_val, seen, recordings, args.normal_protocol)
            test_set = T.select_windows(all_test, seen, recordings, args.normal_protocol)
            ctx = ([w for w in all_train if w["label"] == 0 and w.get("source_file") in new_recs]
                   if args.normal_protocol == "recording" else [])
        new_data = train_new + ctx
        all_seen_train += new_data

        # what this phase trains on
        replay = memory.sample(len(train_new)) if (pi > 0 and uses_replay and args.method != "derpp") else []
        if args.method == "joint":
            train_data = list(all_seen_train)
        else:
            in_replay = {id(w) for w in replay}
            train_data = replay + [w for w in new_data if id(w) not in in_replay]
        natural = memory.natural + T.label_counts(new_data)
        adj = T.logit_adjustment(train_data, natural, seen, args.logit_adjust, device) if pi > 0 else None

        old_model = None
        if pi > 0 and args.method in KD_METHODS:
            old_model = copy.deepcopy(model).eval()
            old_model.use_nme = False
            for p in old_model.parameters():
                p.requires_grad = False
        if pi > 0 and args.method == "der":
            model.add_encoder(device)
        if args.method == "joint" and pi > 0:
            model = DenseGraphIDS(meta, args).to(device)          # retrain from scratch
        model.use_nme = False
        stream = None
        if pi > 0 and args.method == "derpp" and memory.buffer:
            mem = list(memory.buffer)
            stream = ReplayStream(mem, torch.stack([stored_logits[id(w)] for w in mem]), args)

        epochs = args.phase1_epochs if (pi == 0 or args.method == "joint") else args.phase_epochs
        lr = args.lr if (pi == 0 or args.method == "joint") else args.lr * args.phase_lr_mult
        logger.info(f"  Train: new={len(train_new)} context_normal={len(ctx)} replay={len(replay)}"
                    f"{' (stream)' if stream is not None else ''} total={len(train_data)} | "
                    f"epochs={epochs} lr={lr:g} | {model.get_info()}")
        fit(model, train_data, val_set, seen, phase["name"], epochs, lr, args, device, logger, args.method,
            old_model=old_model, old_idx=sorted(seen_before), ewc=ewc, stream=stream, logit_adj=adj,
            new_labels=sorted(new - seen_before))

        # after training: update memory and method state
        memory.add(new_data, pi)
        if ewc is not None:
            ewc.update(model, new_data, args, device)
        if args.method == "derpp":
            missing = [w for w in memory.buffer if id(w) not in stored_logits]
            if missing:
                for w, lg in zip(missing, window_logits(model, missing, args, device)):
                    stored_logits[id(w)] = lg
            keep = {id(w) for w in memory.buffer}
            stored_logits = {k: v for k, v in stored_logits.items() if k in keep}
        if args.method == "icarl":
            model.nme_means = class_means(model, list(memory.buffer), args, device)
            model.use_nme = True
        record(phase, test_set, t0, {"decision": args.method})
        del old_model

    # ------------------------------------------------------------------ summary (as the main trainer)
    learned, forget, plastic = {}, {}, []
    for i, r in enumerate(results):
        for name, v in r["per_attack"].items():
            if name not in learned:
                learned[name] = i
                if i > 0:
                    plastic.append(v)
    final = results[-1]["per_attack"]
    for name, i0 in learned.items():
        hist = [results[j]["per_attack"][name] for j in range(i0, len(results) - 1) if name in results[j]["per_attack"]]
        if hist:
            forget[name] = max(0.0, max(hist) - final[name])
    cl = {"run_name": args.run_name, "method": args.method, "started_at": args.started_at,
          "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "log_name": args.log_name,
          "final_macro_f1": results[-1]["overall"]["macro_f1"],
          "final_normal_fpr": results[-1]["overall"]["normal_fpr"],
          "final_detection_rate": results[-1]["overall"]["detection_rate"],
          "forgetting_max": max(forget.values()) if forget else 0.0,
          "forgetting_max_class": max(forget, key=forget.get) if forget else None,
          "forgetting_mean": float(np.mean(list(forget.values()))) if forget else 0.0,
          "new_class_f1_when_learned": float(np.mean(plastic)) if plastic else None,
          "final_params": results[-1]["params"], "total_minutes": sum(r["minutes"] for r in results)}
    logger.info("\nCL summary: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in cl.items()}))
    logger.info("\nPer-class F1 across phases:")
    for name in [c for c in CLASS_NAMES if c in learned]:
        logger.info(f"  {name:>12s}: " + " -> ".join(f"{r['per_attack'].get(name, 0):.3f}" for r in results))
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(args.save_dir, "cl_summary.json"), "w") as f:
        json.dump(cl, f, indent=2, default=str)
    logger.info(f"\nDone. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
