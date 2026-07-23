"""Minimal LoRA (no external deps) for fine-tuning the CONCH ViT encoder.

We can't rely on `peft` being installed in the training venv, and the integration surface
we need is tiny: wrap selected nn.Linear layers with a frozen base + a trainable low-rank
update  W x + (alpha/r) * (B (A x)).  B is zero-initialised so the adapted model starts
identical to the frozen encoder. Only the LoRA params (and, separately, the MIL head) are
trained; the 86M-param CONCH backbone stays frozen, keeping the trainable set ~1M params.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)   # B stays 0 -> initial delta = 0

    def forward(self, x):
        out = self.base(x)
        delta = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)
        return out + self.scaling * delta


def inject_lora(model, targets=("attn.qkv", "attn.proj"), require="visual.trunk",
                r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Replace matching nn.Linear layers (by full-name suffix, restricted to `require`)
    with LoRALinear, freeze everything else, and unfreeze only the LoRA params.
    Returns the number of adapted layers."""
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if (isinstance(child, nn.Linear)
                    and (require is None or require in full)
                    and any(full.endswith(t) for t in targets)):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                n += 1
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    return n


def lora_state_dict(model):
    """Just the trainable LoRA tensors (small, for checkpointing)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def load_lora_state(model, sd):
    model.load_state_dict(sd, strict=False)
