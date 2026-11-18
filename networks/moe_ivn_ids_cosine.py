"""
moe_ivn_ids_cosine.py — MoE-based IVN IDS with cosine head and continual learning support

Pipeline:
  1. CAN ID Embedding:  id_idx → embed ∈ R^E
  2. Feature concat:    [behavioral_features ‖ embed] ∈ R^{F+E}
  3. MoE Projection:    → s_i (with routing entropy)
  4. Graph Aggregation:  u_i = Σ adj_{ij} s_j
  5. Deviation:         h_i = [s_i ‖ u_i ‖ s_i − u_i]
  6. Readout:           MeanMaxPool → z_W
  7. Classifier:        MLP → ŷ

Continual learning:
  - Entropy monitor detects novel patterns
  - Expert pool expandable via add_expert()
  - Freeze/unfreeze helpers for incremental training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.moe_layer import MoEProjection


class CosineLinear(nn.Module):
    """Cosine similarity classifier (from LUCIR).
    Normalizes both features and weights before computing logits.
    Prevents majority-class bias from weight norm differences.
    """

    def __init__(self, in_dim, num_classes, temperature=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.temperature * torch.mm(x_norm, w_norm.t())


class MoEIVNIDS(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2049,
        num_features: int = 12,
        embed_dim: int = 16,
        num_classes: int = 10,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden: int = 64,
        moe_out_dim: int = 128,
        cls_hidden: int = 128,
        dropout: float = 0.2,
        balance_weight: float = 0.01,
        diversity_weight: float = 0.01,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.moe_out_dim = moe_out_dim
        self.balance_weight = balance_weight
        self.diversity_weight = diversity_weight

        # Stage 1: Learnable CAN ID embedding
        self.id_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Stage 2: MoE projection
        moe_in_dim = num_features + embed_dim
        self.moe = MoEProjection(
            moe_in_dim, expert_hidden, moe_out_dim,
            num_experts, top_k, dropout,
        )
        self.post_moe_norm = nn.LayerNorm(moe_out_dim)

        # Stage 3: Readout + classifier
        dev_dim = moe_out_dim * 3       # [s, u, s-u]
        readout_dim = dev_dim * 2        # MeanMaxPool
        self.readout_norm = nn.LayerNorm(readout_dim)
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, cls_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(cls_hidden, cls_hidden // 2), nn.SiLU(), nn.Dropout(dropout),
        )
        self.cos_head = CosineLinear(cls_hidden // 2, num_classes)

    def encode_window(self, x, adj, id_idx):
        """Encode one window. Returns h, aux_loss, entropy."""
        embed = self.id_embedding(id_idx)                # [N, E]
        x_cat = torch.cat([x, embed], dim=-1)            # [N, F+E]

        s, aux, weights, entropy = self.moe(x_cat)       # [N, D], ...
        s = self.post_moe_norm(s)

        u = torch.mm(adj, s)                             # [N, D]
        h = torch.cat([s, u, s - u], dim=-1)             # [N, 3D]

        return h, aux, entropy, weights

    def forward(self, x_list, adj_list, id_idx_list, return_details=False):
        B = len(x_list)
        sizes = [x.size(0) for x in x_list]

        # Batch all nodes together → one MoE pass instead of B passes
        x_cat = torch.cat(x_list, dim=0)                     # [total_N, F]
        ids_cat = torch.cat(id_idx_list, dim=0)              # [total_N]
        embed_cat = self.id_embedding(ids_cat)                # [total_N, E]
        input_cat = torch.cat([x_cat, embed_cat], dim=-1)    # [total_N, F+E]

        s_cat, aux_loss, weights_cat, entropy_cat = self.moe(input_cat)
        s_cat = self.post_moe_norm(s_cat)

        # Split back per graph for graph-dependent operations
        s_split = torch.split(s_cat, sizes)
        ent_split = torch.split(entropy_cat, sizes)
        w_split = torch.split(weights_cat, sizes) if return_details else [None] * B

        z_list = []
        all_entropy = []
        all_weights = []

        for i in range(B):
            s = s_split[i]
            adj = adj_list[i]
            u = torch.mm(adj, s)                              # [N, D]
            h = torch.cat([s, u, s - u], dim=-1)              # [N, 3D]
            z = torch.cat([h.mean(0), h.max(0)[0]], dim=-1)   # [6D]
            z_list.append(z)
            all_entropy.append(ent_split[i].mean())
            if return_details:
                all_weights.append(w_split[i].detach())

        z = self.readout_norm(torch.stack(z_list))
        features = self.classifier(z)      # hidden features before cosine head
        logits = self.cos_head(features)    # cosine similarity logits

        cls_probs = F.softmax(logits, dim=-1)
        cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-12)).sum(dim=-1)

        result = {
            "logits": logits,
            "features": features,           # for feature distillation
            "aux_loss": aux_loss,
            "diversity_loss": self._diversity_loss(),
            "entropy": torch.stack(all_entropy),       # routing (giữ cho analysis)
            "cls_entropy": cls_entropy,                 # classification (cho novelty)
        }
        if return_details:
            result["routing_weights"] = all_weights
        return result

    def forward_padded(self, x, ids, adj, n, return_details=False):
        """Same computation as forward(), vectorised over the whole batch.

        x [B,N,F], ids [B,N], adj [B,N,N] zero-padded, n [B] real node counts.
        The MoE runs only on real nodes, in the same order as torch.cat(x_list),
        so balance loss, routing and routing entropy match forward().
        """
        B, N, _ = x.shape
        mask = torch.arange(N, device=x.device)[None, :] < n[:, None]      # [B,N]
        inp = torch.cat([x, self.id_embedding(ids.long())], dim=-1)[mask]   # [M, F+E]

        s_v, aux_loss, weights_v, entropy_v = self.moe(inp)
        s_v = self.post_moe_norm(s_v)

        s = s_v.new_zeros(B, N, s_v.size(-1))
        s[mask] = s_v
        u = torch.bmm(adj.to(s.dtype), s)                                    # [B,N,D]
        h = torch.cat([s, u, s - u], dim=-1)                                 # [B,N,3D]

        m = mask.unsqueeze(-1)
        mean = (h * m).sum(dim=1) / n.unsqueeze(-1).to(h.dtype)
        mx = h.masked_fill(~m, float("-inf")).amax(dim=1)
        z = self.readout_norm(torch.cat([mean, mx], dim=-1))

        features = self.classifier(z)
        logits = self.cos_head(features)

        cls_probs = F.softmax(logits, dim=-1)
        cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-12)).sum(dim=-1)
        ent = entropy_v.new_zeros(B, N)
        ent[mask] = entropy_v
        ent = ent.sum(dim=1) / n.to(ent.dtype)

        result = {
            "logits": logits,
            "features": features,
            "aux_loss": aux_loss,
            "diversity_loss": self._diversity_loss(),
            "entropy": ent,
            "cls_entropy": cls_entropy,
        }
        if return_details:
            result["routing_weights"] = list(torch.split(weights_v.detach(), n.tolist()))
        return result

    def compute_loss(self, output, y, criterion):
        cls = criterion(output["logits"], y)
        aux = output["aux_loss"] * self.balance_weight
        div = output["diversity_loss"] * self.diversity_weight
        total = cls + aux + div
        return total, {
            "total": total.item(), "cls": cls.item(),
            "balance": aux.item(), "diversity": div.item(),
        }

    def _diversity_loss(self):
        weights = [e.net[-1].weight.view(-1) for e in self.moe.experts]
        if len(weights) < 2:
            return torch.tensor(0.0, device=weights[0].device)
        W = F.normalize(torch.stack(weights), dim=-1)
        sim = torch.mm(W, W.t())
        M = sim.size(0)
        mask = ~torch.eye(M, dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()

    # ============================================================
    # Continual learning
    # ============================================================

    def expand_expert_pool(self, device=None, dropout=0.1):
        """Add one new expert, return new expert count.
        Note: base experts use the model dropout (0.2 by default) while added
        experts default to 0.1, as in all runs so far. Pass dropout explicitly
        to make them consistent."""
        in_dim = self.num_features + self.embed_dim
        n = self.moe.add_expert(in_dim, 64, self.moe_out_dim, dropout=dropout, device=device)
        print(f"[Expand] Expert pool: {n-1} → {n}")
        return n

    def freeze_experts(self, except_last_n=0):
        """Freeze all experts except the last N (newly added)."""
        total = len(self.moe.experts)
        for i, expert in enumerate(self.moe.experts):
            freeze = i < total - except_last_n
            for p in expert.parameters():
                p.requires_grad = not freeze
        frozen = total - except_last_n
        print(f"[Freeze] {frozen}/{total} experts frozen, {except_last_n} trainable")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_for_expansion(self):
        """Freeze old experts + embedding. Trainable: new expert + router + classifier."""
        for p in self.parameters():
            p.requires_grad = False
        # Unfreeze new expert (last one)
        for p in self.moe.experts[-1].parameters():
            p.requires_grad = True
        # Unfreeze router
        for p in self.moe.router.parameters():
            p.requires_grad = True
        # Unfreeze classifier + cosine head + norms
        for p in self.classifier.parameters():
            p.requires_grad = True
        for p in self.cos_head.parameters():
            p.requires_grad = True
        for p in self.post_moe_norm.parameters():
            p.requires_grad = True
        for p in self.readout_norm.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Freeze] Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")

    def get_info(self):
        total = sum(p.numel() for p in self.parameters())
        expert_p = sum(p.numel() for e in self.moe.experts for p in e.parameters())
        router_p = sum(p.numel() for p in self.moe.router.parameters())
        return {"total": total, "experts": expert_p, "router": router_p,
                "num_experts": len(self.moe.experts)}
