"""
moe_layer.py — Mixture-of-Experts with entropy-based novelty detection

Components:
  Expert:         Small MLP (behavioral primitive learner)
  Router:         Top-k gating with entropy computation
  MoEProjection:  Combines experts + router + auxiliary losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    def __init__(self, in_dim, num_experts, top_k=2, noise_std=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        weights = torch.zeros_like(logits)
        # weights.scatter_(1, topk_idx, topk_weights)
        weights.scatter_(1, topk_idx, topk_weights.to(weights.dtype))
        # Full softmax for entropy computation (not sparse)
        # full_probs = F.softmax(logits, dim=-1)
        full_probs = F.softmax(logits, dim=-1).to(logits.dtype)
        
        return weights, topk_idx, full_probs


class MoEProjection(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts=8,
                 top_k=2, dropout=0.1, noise=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.router = Router(in_dim, num_experts, top_k, noise)
        self.experts = nn.ModuleList([
            Expert(in_dim, hidden_dim, out_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Returns: output [B,D], aux_loss scalar, weights [B,M], entropy [B]
        """
        weights, _, full_probs = self.router(x)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        output = torch.einsum("bm,bmd->bd", weights, expert_outputs)

        aux_loss = self._balance_loss(weights)

        # Per-sample routing entropy: H(π) = -Σ p log p
        entropy = -(full_probs * torch.log(full_probs + 1e-12)).sum(dim=-1)

        return output, aux_loss, weights, entropy

    def _balance_loss(self, weights):
        M = self.num_experts
        f = (weights > 0).float().mean(dim=0)
        p = weights.mean(dim=0)
        return M * (f * p).sum()

    def add_expert(self, in_dim, hidden_dim, out_dim, dropout=0.1, device=None):
        """Dynamically add a new expert to the pool."""
        new_expert = Expert(in_dim, hidden_dim, out_dim, dropout)
        if device:
            new_expert = new_expert.to(device)
        self.experts.append(new_expert)
        self.num_experts += 1

        # Expand router gate
        old_gate = self.router.gate
        new_gate = nn.Linear(old_gate.in_features, self.num_experts, bias=False)
        if device:
            new_gate = new_gate.to(device)
        with torch.no_grad():
            new_gate.weight[:self.num_experts - 1] = old_gate.weight
            nn.init.xavier_uniform_(new_gate.weight[self.num_experts - 1:])
        self.router.gate = new_gate
        self.router.num_experts = self.num_experts

        return self.num_experts
