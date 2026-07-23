"""Pathology tile encoders for REG2026.

Primary: UNI2-h (MahmoodLab/UNI2-h) — ViT-H, 1536-d, top-tier pure-vision pathology
encoder; already accessible and best for our classification-only task. Optional: CONCH
(vision-language) once HF access is granted. Same embed_tiles() works for both via a
uniform forward returning pooled features. Weights bundled into the offline tarball.
"""
from __future__ import annotations
import torch
import numpy as np

# Per-encoder preprocessing normalization. UNI/UNI2-h use ImageNet stats; CONCH
# (conch_ViT-B-16) uses OpenAI-CLIP stats — using the wrong std degrades a frozen
# encoder's features, so keep these matched to each model's training transform.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_OPENAI_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_OPENAI_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
_NORM = {"uni2h": (_IMAGENET_MEAN, _IMAGENET_STD),
         "conch": (_OPENAI_MEAN, _OPENAI_STD),
         "virchow2": (_IMAGENET_MEAN, _IMAGENET_STD)}  # Virchow2 uses ImageNet stats
# Back-compat default for any caller that doesn't set per-encoder norm.
_MEAN, _STD = _IMAGENET_MEAN, _IMAGENET_STD

# Virchow2 tile embedding = concat(class token, mean of 256 patch tokens) = 2*1280 = 2560,
# the representation recommended on the model card (beats cls-only for downstream tasks).
EMBED_DIM = {"uni2h": 1536, "conch": 512, "virchow2": 2560}


def load_encoder(name="uni2h", device="cuda", weights_path=None):
    """Return (callable encoder taking (B,3,224,224) normalized -> (B,D), dim)."""
    if name == "uni2h":
        import timm
        kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                  embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                  mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                  reg_tokens=8, dynamic_img_size=True)
        # Online (extraction): pretrained=True populates the HF cache. Offline (container):
        # set HF_HUB_OFFLINE=1 and bundle the cache snapshot; this same call loads from it.
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw)
        model.eval().to(device)
        model._emb_mean, model._emb_std = _NORM["uni2h"]
        return model, 1536
    elif name == "conch":
        from conch.open_clip_custom import create_model_from_pretrained
        src = weights_path or "hf_hub:MahmoodLab/conch"
        model, _ = create_model_from_pretrained("conch_ViT-B-16", src, device=device)
        model.eval()

        class _Wrap(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                return self.m.encode_image(x, proj_contrast=False, normalize=False)
        wrapped = _Wrap(model).to(device)
        wrapped._emb_mean, wrapped._emb_std = _NORM["conch"]
        return wrapped, 512
    elif name == "virchow2":
        import timm
        from timm.layers import SwiGLUPacked
        # vit_huge_patch14_224, 4 register tokens, SwiGLU MLP, ImageNet norm (per config.json).
        # global_pool="" -> forward returns the full token sequence (B, 1+4+256, 1280).
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.eval().to(device)

        class _Wrap(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                tok = self.m(x)                  # (B, 261, 1280): cls + 4 reg + 256 patches
                cls = tok[:, 0]                  # (B, 1280)
                patch = tok[:, 5:].mean(dim=1)   # (B, 1280) mean over patch tokens (skip reg)
                return torch.cat([cls, patch], dim=-1)  # (B, 2560)
        wrapped = _Wrap(model).to(device)
        wrapped._emb_mean, wrapped._emb_std = _NORM["virchow2"]
        return wrapped, 2560
    raise ValueError(name)


@torch.no_grad()
def embed_tiles(encoder, tiles_uint8: np.ndarray, device="cuda", batch=128, fp16=True):
    """tiles_uint8 (N,224,224,3) uint8 -> (N,D) float32."""
    x = torch.from_numpy(tiles_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    mean = getattr(encoder, "_emb_mean", _MEAN)
    std = getattr(encoder, "_emb_std", _STD)
    x = (x - mean) / std
    out = []
    use_amp = fp16 and device.startswith("cuda")
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                f = encoder(xb)
        else:
            f = encoder(xb)
        out.append(f.float().cpu())
    return torch.cat(out).numpy().astype(np.float32)
