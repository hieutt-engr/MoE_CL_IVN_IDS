"""
moe_ivn_ids_window.py: behavior MoE whose router and experts also see
WINDOW-level behavior (from preprocess_window.py).

Same as BehaviorMoEIDS (v2: null expert, null/active/exclusivity losses,
identity-path dropout), with one change in the behavior path:

    b_i = [ xb_i , A xb_i , xb_i - A xb_i  ||  wb ]
            node behavior (as in v2)          window behavior, shared by all nodes

wb carries patterns that no single node shows, e.g. the +1 ID sweep of a
systematic attack (versus random IDs in fuzzing), several IDs raised at once
in combined spoofing, or a same-ID frame right after the genuine one
(interval). It is ID-agnostic and relative to the vehicle's normal traffic.
Benign nodes inside an attack window still see the attack context; the null
loss keeps them on the null expert.

wb enters only the behavior path (router + experts), not the readout, so the
classifier can only use it through the experts.
"""

import torch
import torch.nn.functional as F

from networks.moe_ivn_ids_behavior import BehaviorMoE, BehaviorMoEIDS


class WindowBehaviorMoEIDS(BehaviorMoEIDS):
    def __init__(self, window_dim=9, behavior_dim=13, num_experts=4, top_k=2,
                 expert_hidden=64, moe_out_dim=128, dropout=0.2, **kw):
        super().__init__(behavior_dim=behavior_dim, num_experts=num_experts, top_k=top_k,
                         expert_hidden=expert_hidden, moe_out_dim=moe_out_dim,
                         dropout=dropout, **kw)
        self.window_dim = window_dim
        # rebuild the behavior MoE with the wider input
        self.moe = BehaviorMoE(3 * behavior_dim + window_dim, expert_hidden, moe_out_dim,
                               num_experts, top_k, dropout)

    def forward_padded(self, x, ids, adj, n, xb=None, wb=None, node_atk=None,
                       return_details=False):
        if xb is None or wb is None:
            raise ValueError("WindowBehaviorMoEIDS needs xb and wb: use data from "
                             "preprocess_window.py")
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

        # behavior path: node behavior + window behavior (no identity information)
        xbp = xb.float().masked_fill(~m, 0.0)
        ab = torch.bmm(adj.to(xbp.dtype), xbp)
        wbn = wb.float()[:, None, :].expand(B, N, wb.size(-1))
        b = torch.cat([xbp, ab, xbp - ab, wbn], dim=-1)[mask]               # [M_nodes, 3Fb+Fw]
        y_raw, weights, full, balance = self.moe(b)
        y_v = self._behavior_out(y_raw, weights)
        y = y_v.new_zeros(B, N, y_v.size(-1))
        y[mask] = y_v

        if self.training and self.identity_drop > 0:
            keep = (torch.rand(B, 1, 1, device=x.device) >= self.identity_drop).to(hs.dtype)
            hs = hs * keep
        if self.branch_off == "identity":
            hs = torch.zeros_like(hs)
        elif self.branch_off == "behavior":
            y = torch.zeros_like(y)

        h = torch.cat([hs, y.to(hs.dtype)], dim=-1)                          # [B,N,4D]
        mean = (h * m).sum(dim=1) / n.unsqueeze(-1).to(h.dtype)
        mx = h.masked_fill(~m, float("-inf")).amax(dim=1)
        z = self.readout_norm(torch.cat([mean, mx], dim=-1))
        features = self.classifier(z)
        logits = self.cos_head(features)

        cls_probs = F.softmax(logits.float(), dim=-1)
        cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-12)).sum(dim=-1)
        node_ent = -(full * torch.log(full + 1e-12)).sum(dim=-1)
        ent = node_ent.new_full((B, N), float("-inf"))
        ent[mask] = node_ent

        out = {
            "logits": logits,
            "features": features,
            "aux_loss": balance,
            "diversity_loss": torch.zeros((), device=x.device),
            "entropy": ent.amax(dim=1),
            "cls_entropy": cls_entropy,
            "node_probs": full,
            "node_graph": torch.arange(B, device=x.device)[:, None].expand(B, N)[mask],
        }
        if node_atk is not None:
            out["node_atk"] = node_atk[mask].float()
        if return_details:
            out["routing_weights"] = list(torch.split(weights.detach(), n.tolist()))
            out.update(node_identity=s_v.detach(), node_behavior_raw=y_raw.detach(),
                       node_behavior=y_v.detach(), node_real_weight=weights[:, 1:].sum(-1).detach())
        return out

    def get_info(self):
        info = super().get_info()
        info["window_dim"] = self.window_dim
        return info
