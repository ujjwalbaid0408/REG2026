"""MIL classifier + CONCH encoder loader for the REG2026 submission container.

Self-contained mirror of reg2026/mil.py and reg2026/encoder.py (CONCH path) so the offline
container has no dependency on the training package. Provides:
  - MILClassifier / GatedAttentionMIL: the trained aggregator + organ/dx heads, with the
    hierarchical organ-conditioned dx mask.
  - load_encoder(model_path, device): loads the bundled CONCH encoder, returning a callable
    that maps (B,3,224,224) tiles in [0,1] to (B,512) embeddings (CLIP-normalized internally).
  - dx_label_to_organ_dx: maps a "<organ>||<diagnosis>" label back to (organ, dx-or-None).
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

# OpenAI-CLIP normalization (CONCH's image tower is CLIP-initialized).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim, hidden, attn_dim, dropout):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.attn_V = nn.Linear(hidden, attn_dim)
        self.attn_U = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, x, mask=None):
        h = self.proj(x)
        a = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)
        pooled = torch.bmm(a.unsqueeze(1), h).squeeze(1)
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
        if c.get("dx_organ") is not None:
            self.register_buffer("dx_organ", torch.tensor(c["dx_organ"], dtype=torch.long))
        else:
            self.dx_organ = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def masked_dx_logits(self, dx_logits, organ_idx):
        if self.dx_organ is None:
            return dx_logits
        valid = self.dx_organ.unsqueeze(0) == organ_idx.unsqueeze(1)
        return dx_logits.masked_fill(~valid, -1e4)

    def forward(self, x, mask=None):
        pooled, a = self.mil(x, mask)
        z = self.trunk(pooled)
        return {"organ": self.head_organ(z), "dx": self.head_dx(z)}


def dx_label_to_organ_dx(dx_label):
    organ, _, dx = dx_label.partition("||")
    return organ, (None if dx == "other" else dx)


def load_encoder(model_path, device):
    """Load the bundled CONCH encoder. Returns a callable: (B,3,224,224) in [0,1] -> (B,512)."""
    from conch.open_clip_custom import create_model_from_pretrained
    model_path = Path(model_path)
    # CONCH weights are bundled in the model tarball; accept a few common layouts.
    cands = [model_path / "conch" / "pytorch_model.bin",
             model_path / "conch_weights" / "pytorch_model.bin",
             model_path / "pytorch_model.bin"]
    ckpt = next((str(p) for p in cands if p.exists()), "hf_hub:MahmoodLab/conch")
    model, _ = create_model_from_pretrained("conch_ViT-B-16", ckpt, device=device)
    model.eval()
    mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1).to(device)

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        @torch.no_grad()
        def forward(self, x):                      # x in [0,1], (B,3,224,224)
            x = (x - mean) / std
            return self.m.encode_image(x, proj_contrast=False, normalize=False)

    return _Wrap(model).to(device)
