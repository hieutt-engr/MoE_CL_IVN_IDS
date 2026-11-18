"""
dense_ivn_ids.py — Dense baseline for IVN IDS (no MoE)

Same pipeline as MoE version but replaces the MoE layer with a single MLP
of comparable parameter count. Used for ablation: does MoE expansion matter,
or does replay alone suffice?

Pipeline:
  1. CAN ID Embedding:  id_idx → embed ∈ R^E
  2. Feature concat:    [behavioral_features ‖ embed] ∈ R^{F+E}
  3. Dense Projection:  MLP → s_i ∈ R^D  (replaces MoE)
  4. Graph Aggregation: u_i = Σ adj_{ij} s_j
  5. Deviation:         h_i = [s_i ‖ u_i ‖ s_i − u_i]
  6. Readout:           MeanMaxPool → z_W
  7. Classifier:        MLP → ŷ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseIVNIDS(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2049,
        num_features: int = 12,
        embed_dim: int = 16,
        num_classes: int = 10,
        dense_hidden: int = 389,
        out_dim: int = 128,
        cls_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.out_dim = out_dim

        # Stage 1: CAN ID embedding (same as MoE version)
        self.id_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Stage 2: Dense projection (replaces MoE)
        in_dim = num_features + embed_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, dense_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dense_hidden, out_dim),
        )
        self.post_proj_norm = nn.LayerNorm(out_dim)

        # Stage 3: Readout + classifier (same as MoE version)
        dev_dim = out_dim * 3
        readout_dim = dev_dim * 2
        self.readout_norm = nn.LayerNorm(readout_dim)
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

        s_cat = self.projection(input_cat)
        s_cat = self.post_proj_norm(s_cat)

        s_split = torch.split(s_cat, sizes)
        z_list = []

        for i in range(B):
            s = s_split[i]
            adj = adj_list[i]
            u = torch.mm(adj, s)
            h = torch.cat([s, u, s - u], dim=-1)
            z = torch.cat([h.mean(0), h.max(0)[0]], dim=-1)
            z_list.append(z)

        z = self.readout_norm(torch.stack(z_list))
        logits = self.classifier(z)

        cls_probs = F.softmax(logits, dim=-1)
        cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-12)).sum(dim=-1)

        result = {
            "logits": logits,
            "aux_loss": torch.tensor(0.0, device=logits.device),
            "diversity_loss": torch.tensor(0.0, device=logits.device),
            "entropy": torch.zeros(B, device=logits.device),
            "cls_entropy": cls_entropy,
        }
        return result

    def forward_padded(self, x, ids, adj, n, return_details=False):
        """Same computation as forward(), vectorised over the whole batch.
        x [B,N,F], ids [B,N], adj [B,N,N] zero-padded, n [B] real node counts."""
        B, N, _ = x.shape
        mask = torch.arange(N, device=x.device)[None, :] < n[:, None]
        inp = torch.cat([x, self.id_embedding(ids.long())], dim=-1)[mask]
        s_v = self.post_proj_norm(self.projection(inp))

        s = s_v.new_zeros(B, N, s_v.size(-1))
        s[mask] = s_v
        u = torch.bmm(adj.to(s.dtype), s)
        h = torch.cat([s, u, s - u], dim=-1)
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(dim=1) / n.unsqueeze(-1).to(h.dtype)
        mx = h.masked_fill(~m, float("-inf")).amax(dim=1)
        z = self.readout_norm(torch.cat([mean, mx], dim=-1))
        logits = self.classifier(z)

        cls_probs = F.softmax(logits, dim=-1)
        cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-12)).sum(dim=-1)
        return {
            "logits": logits,
            "aux_loss": torch.tensor(0.0, device=logits.device),
            "diversity_loss": torch.tensor(0.0, device=logits.device),
            "entropy": torch.zeros(B, device=logits.device),
            "cls_entropy": cls_entropy,
        }

    def compute_loss(self, output, y, criterion):
        loss = criterion(output["logits"], y)
        return loss, {"total": loss.item(), "cls": loss.item(),
                      "balance": 0.0, "diversity": 0.0}

    def get_info(self):
        total = sum(p.numel() for p in self.parameters())
        proj_p = sum(p.numel() for p in self.projection.parameters())
        return {"total": total, "projection": proj_p, "type": "dense"}
