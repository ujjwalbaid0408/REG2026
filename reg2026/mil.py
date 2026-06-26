"""Gated-attention MIL multi-task head for REG2026.

Operates on precomputed per-tile foundation-model embeddings (CONCH 512-d) -> attention
pool over tiles -> shared trunk -> two classification heads: organ (10-way) and dx
(<organ>||<diagnosis> bucketed, rare folded to <organ>||other). Predicting (organ, dx)
drives the workflow score toward the ~0.89 oracle ceiling; everything downstream is
deterministic template lookup (see reg2026/templates.py).

Inference contract (src/common/predictor.py): MILClassifier.from_config(cfg),
load_state_dict, then forward(feats[B,N,D], mask=None) -> {"organ": logits, "dx": logits}.
Kept dependency-light (torch only) so it can be mirrored into src/common/mil.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    """Ilse et al. (2018) gated attention pooling with padding mask support."""

    def __init__(self, in_dim, hidden, attn_dim, dropout):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.attn_V = nn.Linear(hidden, attn_dim)
        self.attn_U = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, x, mask=None):
        # x: (B, N, in_dim) ; mask: (B, N) bool, True=valid (None -> all valid)
        h = self.proj(x)                                  # (B, N, hidden)
        a = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))  # (B,N,1)
        a = a.squeeze(-1)                                 # (B, N)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)                       # (B, N)
        pooled = torch.bmm(a.unsqueeze(1), h).squeeze(1)  # (B, hidden)
        return pooled, a


class MILClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        c = self.config
        self.mil = GatedAttentionMIL(c["in_dim"], c["hidden"], c["attn_dim"], c["dropout"])
        self.trunk = nn.Sequential(nn.Dropout(c["dropout"]),
                                    nn.Linear(c["hidden"], c["hidden"]), nn.GELU(),
                                    nn.Dropout(c["dropout"]))
        self.head_organ = nn.Linear(c["hidden"], c["n_organ"])
        self.head_dx = nn.Linear(c["hidden"], c["n_dx"])
        # Hierarchical head: dx_organ[j] = organ index that diagnosis class j belongs to.
        # When present, dx logits are masked to the (predicted/true) organ's classes, turning
        # one 77-way problem into per-organ sub-problems and guaranteeing organ/dx consistency.
        if c.get("dx_organ") is not None:
            self.register_buffer("dx_organ", torch.tensor(c["dx_organ"], dtype=torch.long))
        else:
            self.dx_organ = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def masked_dx_logits(self, dx_logits, organ_idx):
        """Restrict dx logits to classes belonging to organ_idx (hierarchical head).
        organ_idx: (B,) long. Returns dx_logits with out-of-organ classes set to -1e4."""
        if self.dx_organ is None:
            return dx_logits
        valid = self.dx_organ.unsqueeze(0) == organ_idx.unsqueeze(1)   # (B, n_dx)
        return dx_logits.masked_fill(~valid, -1e4)

    def forward(self, x, mask=None, return_attn=False):
        pooled, a = self.mil(x, mask)
        z = self.trunk(pooled)
        out = {"organ": self.head_organ(z), "dx": self.head_dx(z)}
        if return_attn:
            out["attn"] = a
        return out
