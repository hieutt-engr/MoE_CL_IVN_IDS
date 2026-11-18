"""
moe_ivn_ids_behavior.py: MoE whose experts specialise on attack BEHAVIOR, not CAN ID.

Why a new architecture. In moe_ivn_ids_cosine the router reads
[node features || ID embedding]; 16 of its 28 inputs are the ID embedding and
the raw features themselves (absolute byte means, inter-arrival times) are
ID-specific, so routing follows identity. The experts also sit BEFORE graph
aggregation and never see the relational deviation (s - u), where most attack
behavior shows up.

Design:
  identity path (backbone, the "normal traffic model"):
      h0 = MLP([x || emb(id)]) ; u = A h0 ; hs = [h0, u, h0 - u]
      trained in phase 1; with --train_scope backbone it keeps adapting in later
      phases. Nothing forces it to model normal traffic: it is an ID-aware branch
      trained by the classification loss (see --mask_unknown_ids)
  behavior path (router + experts never see the ID):
      xb = per-ID z-scores against the normal profile (+ unknown-ID flag),
           from preprocess_behavior.py; normal traffic gives ~0 for every ID
      b  = [xb, A xb, xb - A xb]                (own value + relational deviation)
      router over 1 + M choices: index 0 is a NULL expert (output 0) that absorbs
      normal nodes, indices 1..M are behavior experts
      y  = sum_k w_k E_k(b)
  readout: masked mean/max of [hs || y] -> MLP -> cosine head

Auxiliary losses (extra_loss):
  null loss         nodes WITHOUT attack frames go to the null expert
  active loss       nodes WITH attack frames must NOT go to the null expert
                    (frame-level labels from preprocess_behavior.py; without them
                    the null loss falls back to all nodes of normal windows)
  exclusivity loss  during an expansion, nodes of windows from OLD classes should
                    not use the new expert (keeps old behaviors on old experts)

Identity-path dropout: during training the identity readout of a window is
zeroed with probability identity_drop, so the classifier cannot solve every
window from the backbone alone and has to use the behavior experts. Version 1
(without the active loss and this dropout) routed 99% of attacked nodes to the
null expert: the backbone did all the work and the experts stayed idle.

Novelty signal: per-node routing entropy over the 1 + M choices; a window's
score is the maximum over its nodes (attacks touch few nodes, a mean would
dilute them). Normal nodes go confidently to null, known behaviors confidently
to their expert, unseen behaviors are routed uncertainly.

Only the vectorised forward_padded() is implemented (needs the fast data path).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.moe_ivn_ids_cosine import CosineLinear
from networks.moe_layer import Expert


class BehaviorRouter(nn.Module):
    def __init__(self, in_dim, num_choices, top_k=2, noise_std=0.1):
        super().__init__()
        self.top_k = min(top_k, num_choices)
        self.noise_std = noise_std
        self.gate = nn.Linear(in_dim, num_choices, bias=True)

    def forward(self, b):
        logits = self.gate(b).float()
        full = F.softmax(logits, dim=-1)                  # clean distribution (entropy, losses)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        val, idx = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.zeros_like(logits).scatter_(1, idx, F.softmax(val, dim=-1))
        return weights, full


class BehaviorMoE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        self.in_dim, self.hidden_dim, self.out_dim = in_dim, hidden_dim, out_dim
        self.router = BehaviorRouter(in_dim, 1 + num_experts, top_k)
        self.experts = nn.ModuleList([Expert(in_dim, hidden_dim, out_dim, dropout)
                                      for _ in range(num_experts)])

    @property
    def num_experts(self):
        return len(self.experts)

    def forward(self, b):
        weights, full = self.router(b)                    # [M_nodes, 1+M]
        outs = torch.stack([e(b) for e in self.experts], dim=1)   # [M_nodes, M, D]
        y = torch.einsum("nm,nmd->nd", weights[:, 1:].to(outs.dtype), outs)
        w_real = weights[:, 1:]
        f = (w_real > 0).float().mean(0)
        p = w_real.mean(0)
        balance = self.num_experts * (f * p).sum()
        return y, weights, full, balance

    def add_expert(self, dropout=0.1, device=None):
        new = Expert(self.in_dim, self.hidden_dim, self.out_dim, dropout)
        old = self.router.gate
        gate = nn.Linear(old.in_features, old.out_features + 1, bias=True)
        if device is not None:
            new, gate = new.to(device), gate.to(device)
        with torch.no_grad():
            gate.weight[:-1] = old.weight
            gate.bias[:-1] = old.bias
            nn.init.xavier_uniform_(gate.weight[-1:])
            gate.bias[-1] = 0.0
        self.experts.append(new)
        self.router.gate = gate
        return self.num_experts


class BehaviorMoEIDS(nn.Module):
    needs_behavior = True

    def __init__(self, vocab_size=2049, num_features=12, behavior_dim=13, embed_dim=16,
                 num_classes=10, num_experts=4, top_k=2, expert_hidden=64, moe_out_dim=128,
                 cls_hidden=128, dropout=0.2, balance_weight=0.01,
                 null_weight=0.1, excl_weight=0.1, active_weight=0.5, identity_drop=0.3,
                 null_scale=False, mask_unknown_ids=False):
        super().__init__()
        # null_scale: behavior output = LN(sum_e w_e E_e) * sum_e w_e (real experts only),
        #   so a node routed entirely to the null expert contributes exactly 0. Without it,
        #   LayerNorm is scale invariant and restores the second expert's output.
        # mask_unknown_ids: CAN IDs without a normal profile (unknown flag = last column of
        #   xb) use the padding embedding (index 0, no gradient), so the identity branch
        #   cannot memorise "this ID only appears in attacks"; unknown-ness reaches the
        #   classifier only through the behavior branch.
        self.null_scale, self.mask_unknown_ids = null_scale, mask_unknown_ids
        self.branch_off = None                 # diagnostics: None | "identity" | "behavior"
        D = moe_out_dim
        self.balance_weight = balance_weight
        self.diversity_weight = 0.0
        self.null_weight, self.excl_weight = null_weight, excl_weight
        self.active_weight, self.identity_drop = active_weight, identity_drop
        self.excl_new_labels = None            # set by the trainer during an expansion
        self.dropout = dropout

        # identity path
        self.id_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.backbone = nn.Sequential(
            nn.Linear(num_features + embed_dim, 2 * D), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(2 * D, D))
        self.post_norm = nn.LayerNorm(D)

        # behavior path
        self.moe = BehaviorMoE(3 * behavior_dim, expert_hidden, D, num_experts, top_k, dropout)
        self.behavior_norm = nn.LayerNorm(D)

        # readout + head
        readout_dim = 6 * D + 2 * D
        self.readout_norm = nn.LayerNorm(readout_dim)
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, cls_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(cls_hidden, cls_hidden // 2), nn.SiLU(), nn.Dropout(dropout))
        self.cos_head = CosineLinear(cls_hidden // 2, num_classes)

    # ------------------------------------------------------------------ helpers
    def _identity_ids(self, ids, xb):
        ids = ids.long()
        if self.mask_unknown_ids and xb is not None:
            ids = ids.masked_fill(xb[..., -1] > 0.5, 0)
        return ids

    def _behavior_out(self, y_raw, weights):
        y = self.behavior_norm(y_raw)
        if self.null_scale:
            y = y * weights[:, 1:].sum(-1, keepdim=True).to(y.dtype)
        return y

    # ------------------------------------------------------------------ forward
    def forward(self, *a, **k):
        raise NotImplementedError("BehaviorMoEIDS needs the fast data path (forward_padded)")

    def forward_padded(self, x, ids, adj, n, xb=None, node_atk=None, return_details=False):
        if xb is None:
            raise ValueError("BehaviorMoEIDS needs behavior features: use data from "
                             "preprocess_behavior.py")
        B, N, _ = x.shape
        mask = torch.arange(N, device=x.device)[None, :] < n[:, None]      # [B,N]
        m = mask.unsqueeze(-1)

        # identity path
        inp = torch.cat([x, self.id_embedding(self._identity_ids(ids, xb))], dim=-1)[mask]
        s_v = self.post_norm(self.backbone(inp))
        s = s_v.new_zeros(B, N, s_v.size(-1))
        s[mask] = s_v
        u = torch.bmm(adj.to(s.dtype), s)
        hs = torch.cat([s, u, s - u], dim=-1)                                # [B,N,3D]

        # behavior path (no identity information)
        xbp = xb.float().masked_fill(~m, 0.0)
        ab = torch.bmm(adj.to(xbp.dtype), xbp)
        b = torch.cat([xbp, ab, xbp - ab], dim=-1)[mask]                     # [M_nodes, 3Fb]
        y_raw, weights, full, balance = self.moe(b)
        y_v = self._behavior_out(y_raw, weights)
        y = y_v.new_zeros(B, N, y_v.size(-1))
        y[mask] = y_v

        # identity-path dropout (training only): whole windows lose the backbone readout
        if self.training and self.identity_drop > 0:
            keep = (torch.rand(B, 1, 1, device=x.device) >= self.identity_drop).to(hs.dtype)
            hs = hs * keep
        if self.branch_off == "identity":
            hs = torch.zeros_like(hs)
        elif self.branch_off == "behavior":
            y = torch.zeros_like(y)

        # readout
        h = torch.cat([hs, y.to(hs.dtype)], dim=-1)                          # [B,N,4D]
        mean = (h * m).sum(dim=1) / n.unsqueeze(-1).to(h.dtype)
        mx = h.masked_fill(~m, float("-inf")).amax(dim=1)
        z = self.readout_norm(torch.cat([mean, mx], dim=-1))
        features = self.classifier(z)
        logits = self.cos_head(features)

        cls_probs = F.softmax(logits.float(), dim=-1)
        cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-12)).sum(dim=-1)

        node_ent = -(full * torch.log(full + 1e-12)).sum(dim=-1)            # [M_nodes]
        ent = node_ent.new_full((B, N), float("-inf"))
        ent[mask] = node_ent
        window_route_entropy = ent.amax(dim=1)

        node_graph = torch.arange(B, device=x.device)[:, None].expand(B, N)[mask]
        out = {
            "logits": logits,
            "features": features,
            "aux_loss": balance,
            "diversity_loss": torch.zeros((), device=x.device),
            "entropy": window_route_entropy,      # behavior routing novelty score
            "cls_entropy": cls_entropy,
            "node_probs": full,                   # [M_nodes, 1+M], index 0 = null
            "node_graph": node_graph,             # window index of each node
        }
        if node_atk is not None:
            out["node_atk"] = node_atk[mask].float()
        if return_details:
            out["routing_weights"] = list(torch.split(weights.detach(), n.tolist()))
            out.update(node_identity=s_v.detach(), node_behavior_raw=y_raw.detach(),
                       node_behavior=y_v.detach(), node_real_weight=weights[:, 1:].sum(-1).detach())
        return out

    # ------------------------------------------------------------------ losses
    def extra_loss(self, out, y):
        probs, ny = out["node_probs"], y[out["node_graph"]]
        loss = probs.new_zeros(())
        p_null = probs[:, 0]
        if "node_atk" in out:                    # frame-level ground truth
            attacked = out["node_atk"] > 0
            benign = ~attacked
        else:                                    # fallback: window labels only
            attacked = torch.zeros_like(ny, dtype=torch.bool)
            benign = ny == 0
        if self.null_weight > 0 and benign.any():
            loss = loss - self.null_weight * torch.log(p_null[benign] + 1e-6).mean()
        if self.active_weight > 0 and attacked.any():
            loss = loss - self.active_weight * torch.log(1 - p_null[attacked] + 1e-6).mean()
        if self.excl_weight > 0 and self.excl_new_labels is not None:
            old = ~torch.isin(ny, self.excl_new_labels)
            if old.any():
                loss = loss + self.excl_weight * probs[old, -1].mean()
        return loss

    def compute_loss(self, output, y, criterion):
        cls = criterion(output["logits"], y)
        total = cls + output["aux_loss"] * self.balance_weight + self.extra_loss(output, y)
        return total, {"total": total.item(), "cls": cls.item()}

    # ------------------------------------------------------------------ continual
    def expand_expert_pool(self, device=None, dropout=None):
        k = self.moe.add_expert(dropout=self.dropout if dropout is None else dropout,
                                device=device)
        print(f"[Expand] Behavior experts: {k - 1} → {k}")
        return k

    def freeze_for_expansion(self):
        """Trainable: new expert, router, readout norm, classifier, cosine head.
        Frozen: identity path (embedding, backbone, post_norm), old experts,
        behavior_norm. Rows of the router that belong to old choices can be
        pinned by the trainer (--freeze_old_gate)."""
        for p in self.parameters():
            p.requires_grad = False
        for mod in (self.moe.experts[-1], self.moe.router, self.readout_norm,
                    self.classifier, self.cos_head):
            for p in mod.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Freeze] Trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_info(self):
        total = sum(p.numel() for p in self.parameters())
        expert_p = sum(p.numel() for e in self.moe.experts for p in e.parameters())
        router_p = sum(p.numel() for p in self.moe.router.parameters())
        backbone_p = sum(p.numel() for p in self.backbone.parameters())
        return {"total": total, "backbone": backbone_p, "experts": expert_p,
                "router": router_p, "num_experts": self.moe.num_experts}
