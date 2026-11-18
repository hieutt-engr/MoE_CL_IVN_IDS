"""
moe_ivn_ids.py — MoE-based IVN IDS with optional architectural improvements

Three toggleable improvements over v1, each testable independently:

  A. use_residual:       Skip connection around MoE (frozen when expand)
  B. light_classifier:   2-layer classifier instead of 3 (reduces drift)
  C. attention_readout:   Anomaly-aware attention pooling (frozen when expand)

Default (all False) = identical behavior to v1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.moe_layer import MoEProjection


class AttentionReadout(nn.Module):
    """Anomaly-aware attention readout.

    Nodes that deviate more from neighbors get higher attention weight.
    Frozen when expanding: the principle "anomalous nodes matter more"
    holds for new attacks too.
    """

    def __init__(self, dev_dim, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dev_dim, dev_dim // 4), nn.SiLU(),
            nn.Linear(dev_dim // 4, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        """h: [N, 3D] deviation features. Returns: [3D] graph-level."""
        w = self.score(h)
        w = F.softmax(w, dim=0)
        w = self.dropout(w)
        return (w * h).sum(dim=0)


class MoEIVNIDSv2(nn.Module):
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
        # === Toggleable improvements ===
        use_residual: bool = False,
        light_classifier: bool = False,
        attention_readout: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.moe_out_dim = moe_out_dim
        self.balance_weight = balance_weight
        self.diversity_weight = diversity_weight
        self.use_residual = use_residual
        self.light_classifier = light_classifier
        self.attention_readout = attention_readout

        # Stage 1: CAN ID embedding
        self.id_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Stage 2: MoE projection
        moe_in_dim = num_features + embed_dim
        self.moe = MoEProjection(
            moe_in_dim, expert_hidden, moe_out_dim,
            num_experts, top_k, dropout,
        )

        # [A] Residual: frozen when expand, preserves input signal
        if self.use_residual:
            self.residual_proj = nn.Linear(moe_in_dim, moe_out_dim)

        self.post_moe_norm = nn.LayerNorm(moe_out_dim)

        # Deviation dim
        dev_dim = moe_out_dim * 3  # [s, u, s-u]

        # [C] Attention readout: frozen when expand
        if self.attention_readout:
            self.readout_attn = AttentionReadout(dev_dim, dropout)
            readout_dim = dev_dim         # 384
        else:
            readout_dim = dev_dim * 2     # 768 (MeanMaxPool)

        self.readout_norm = nn.LayerNorm(readout_dim)

        # [B] Classifier: light (2 layers) or standard (3 layers)
        if self.light_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(readout_dim, cls_hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(cls_hidden, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(readout_dim, cls_hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(cls_hidden, cls_hidden // 2), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(cls_hidden // 2, num_classes),
            )

    def forward(self, x_list, adj_list, id_idx_list, return_details=False):
        B = len(x_list)
        sizes = [x.size(0) for x in x_list]

        x_cat = torch.cat(x_list, dim=0)
        ids_cat = torch.cat(id_idx_list, dim=0)
        embed_cat = self.id_embedding(ids_cat)
        input_cat = torch.cat([x_cat, embed_cat], dim=-1)

        s_cat, aux_loss, weights_cat, entropy_cat = self.moe(input_cat)

        # [A] Residual
        if self.use_residual:
            s_cat = s_cat + self.residual_proj(input_cat)

        s_cat = self.post_moe_norm(s_cat)

        s_split = torch.split(s_cat, sizes)
        ent_split = torch.split(entropy_cat, sizes)
        w_split = torch.split(weights_cat, sizes) if return_details else [None] * B

        z_list = []
        all_entropy = []
        all_weights = []

        for i in range(B):
            s = s_split[i]
            adj = adj_list[i]
            u = torch.mm(adj, s)
            h = torch.cat([s, u, s - u], dim=-1)

            # [C] Readout
            if self.attention_readout:
                z = self.readout_attn(h)
            else:
                z = torch.cat([h.mean(0), h.max(0)[0]], dim=-1)

            z_list.append(z)
            all_entropy.append(ent_split[i].mean())
            if return_details:
                all_weights.append(w_split[i].detach())

        z = self.readout_norm(torch.stack(z_list))
        logits = self.classifier(z)

        cls_probs = F.softmax(logits, dim=-1)
        cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-12)).sum(dim=-1)

        result = {
            "logits": logits,
            "aux_loss": aux_loss,
            "diversity_loss": self._diversity_loss(),
            "entropy": torch.stack(all_entropy),
            "cls_entropy": cls_entropy,
        }
        if return_details:
            result["routing_weights"] = all_weights
        return result

    def forward_padded(self, x, ids, adj, n, return_details=False):
        """Same computation as forward(), vectorised over the whole batch.

        x [B,N,F], ids [B,N], adj [B,N,N] zero-padded, n [B] real node counts.
        The MoE runs only on real nodes (same order as torch.cat(x_list)), so
        balance loss, routing and entropy match the list-based forward.
        """
        B, N, _ = x.shape
        mask = torch.arange(N, device=x.device)[None, :] < n[:, None]      # [B,N]
        inp = torch.cat([x, self.id_embedding(ids.long())], dim=-1)[mask]   # [M, F+E]

        s_v, aux_loss, weights_v, entropy_v = self.moe(inp)
        if self.use_residual:
            s_v = s_v + self.residual_proj(inp)
        s_v = self.post_moe_norm(s_v)

        s = s_v.new_zeros(B, N, s_v.size(-1))
        s[mask] = s_v
        u = torch.bmm(adj.to(s.dtype), s)
        h = torch.cat([s, u, s - u], dim=-1)                                 # [B,N,3D]
        m = mask.unsqueeze(-1)

        if self.attention_readout:
            w = self.readout_attn.score(h).masked_fill(~m, float("-inf"))
            w = self.readout_attn.dropout(F.softmax(w, dim=1))
            z = (w * h).sum(dim=1)
        else:
            mean = (h * m).sum(dim=1) / n.unsqueeze(-1).to(h.dtype)
            mx = h.masked_fill(~m, float("-inf")).amax(dim=1)
            z = torch.cat([mean, mx], dim=-1)

        z = self.readout_norm(z)
        logits = self.classifier(z)

        cls_probs = F.softmax(logits, dim=-1)
        cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-12)).sum(dim=-1)
        ent = entropy_v.new_zeros(B, N)
        ent[mask] = entropy_v
        ent = ent.sum(dim=1) / n.to(ent.dtype)

        result = {
            "logits": logits,
            "aux_loss": aux_loss,
            "diversity_loss": self._diversity_loss(),
            "entropy": ent,
            "cls_entropy": cls_entropy,
        }
        if return_details:
            result["routing_weights"] = list(
                torch.split(weights_v.detach(), n.tolist()))
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

    def expand_expert_pool(self, device=None):
        in_dim = self.num_features + self.embed_dim
        n = self.moe.add_expert(in_dim, 64, self.moe_out_dim, device=device)
        print(f"[Expand] Expert pool: {n-1} → {n}")
        return n

    def freeze_experts(self, except_last_n=0):
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
        """Freeze old experts + embedding + residual_proj + attention_readout.
        Trainable: new expert + router + classifier + norms.
        """
        for p in self.parameters():
            p.requires_grad = False

        # New expert
        for p in self.moe.experts[-1].parameters():
            p.requires_grad = True
        # Router
        for p in self.moe.router.parameters():
            p.requires_grad = True
        # Classifier + norms
        for p in self.classifier.parameters():
            p.requires_grad = True
        for p in self.post_moe_norm.parameters():
            p.requires_grad = True
        for p in self.readout_norm.parameters():
            p.requires_grad = True

        # residual_proj: FROZEN (preserves input signal)
        # readout_attn: FROZEN (anomaly logic generalizes)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen_extra = []
        if self.use_residual:
            frozen_extra.append("residual_proj")
        if self.attention_readout:
            frozen_extra.append("readout_attn")
        print(f"[Freeze] Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
        if frozen_extra:
            print(f"[Freeze] Also frozen: {', '.join(frozen_extra)}")

    def get_info(self):
        total = sum(p.numel() for p in self.parameters())
        expert_p = sum(p.numel() for e in self.moe.experts for p in e.parameters())
        router_p = sum(p.numel() for p in self.moe.router.parameters())
        cls_p = sum(p.numel() for p in self.classifier.parameters())
        info = {
            "total": total, "experts": expert_p, "router": router_p,
            "classifier": cls_p, "num_experts": len(self.moe.experts),
            "flags": f"residual={self.use_residual} light_cls={self.light_classifier} attn_readout={self.attention_readout}",
        }
        if self.use_residual:
            info["residual_proj"] = sum(
                p.numel() for p in self.residual_proj.parameters())
        if self.attention_readout:
            info["readout_attn"] = sum(
                p.numel() for p in self.readout_attn.parameters())
        return info
