"""Pluggable WSI -> (organ, dx, gradings) predictor for REG2026 inference.

Design: the container always emits valid output. If a trained classifier is present in
MODEL_PATH it is used; otherwise we fall back to the prior baseline (global modal
template). This lets the SAME container run before and after the model is trained.

The trained model contract (when present at MODEL_PATH):
  - encoder: a TorchScript / timm pathology foundation encoder producing per-tile
    feature vectors (loaded by the embedding pipeline; bundled offline in the tarball)
  - mil_head.pt: state dict for the MIL aggregator + multi-task heads
  - label_maps.json: {"organ": [...], "dx": [...], ...} index->string
We keep this module dependency-light; heavy imports happen only inside _load_model.
"""
from __future__ import annotations
import json
from pathlib import Path

_STATE = {"loaded": False, "model": None, "labels": None, "encoder": None,
          "encoder2": None, "fusion": False, "device": None, "spec": None}


def _try_load_model(model_path: Path):
    """Attempt to load the trained classifier. Return True on success.

    Supports both CONCH-only (in_dim 512) and CONCH+UNI2-h fusion (in_dim 2048) heads;
    for fusion we also load the UNI2-h encoder so per-tile features can be concatenated
    [CONCH | UNI2-h] exactly as in training.
    """
    head = model_path / "mil_head.pt"
    labels = model_path / "label_maps.json"
    if not (head.exists() and labels.exists()):
        return False
    try:
        import torch
        from .mil import MILClassifier, load_encoder, load_uni2h_encoder
        _STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        _STATE["labels"] = json.loads(labels.read_text())
        ck = torch.load(head, map_location=_STATE["device"])
        model = MILClassifier.from_config(ck["config"])
        model.load_state_dict(ck["state_dict"])
        model.to(_STATE["device"]).eval()
        _STATE["model"] = model
        _STATE["fusion"] = int(ck["config"].get("in_dim", 512)) == 2048
        _STATE["encoder"] = load_encoder(model_path, _STATE["device"])      # CONCH (512)
        if _STATE["fusion"]:
            _STATE["encoder2"] = load_uni2h_encoder(model_path, _STATE["device"])  # UNI2-h (1536)
            print("[predictor] fusion model loaded (CONCH+UNI2-h -> 2048)")
        # Optional prostate tumor-detection specialist (overrides the main head's
        # prostate No-tumor<->Acinar call). Fully optional: absence leaves spec=None.
        try:
            spec_path = model_path / "spec_head.pt"
            if _STATE["fusion"] and spec_path.exists():
                from .mil import ProstateSpecialist
                sck = torch.load(spec_path, map_location=_STATE["device"])
                spec = ProstateSpecialist(in_dim=int(sck.get("in_dim", 2048)))
                spec.load_state_dict(sck["state_dict"])
                spec.to(_STATE["device"]).eval()
                _STATE["spec"] = spec
                print("[predictor] prostate specialist loaded")
        except Exception as e:
            print(f"[predictor] prostate specialist not loaded ({e}); skipping override")
            _STATE["spec"] = None
        return True
    except Exception as e:  # never crash inference on load failure -> use baseline
        print(f"[predictor] model load failed ({e}); using baseline prior")
        _STATE["model"] = None
        return False


def _baseline_prediction(templates):
    """No trained model: predict nothing -> apply_template falls back to global modal."""
    return {"organ": None}


def predict_fields(wsi_path, model_path: Path, templates) -> dict:
    """Return a dict of predicted answer fields: at least {'organ': ...}, ideally
    {'organ','dx', gradings...}. Used by apply_template()."""
    if not _STATE["loaded"]:
        _STATE["loaded"] = True
        _try_load_model(Path(model_path))

    if _STATE["model"] is None:
        return _baseline_prediction(templates)

    # --- trained-model path ---
    import torch
    from .wsi import sample_tiles
    from .mil import dx_label_to_organ_dx
    tiles = sample_tiles(str(wsi_path))  # (N,224,224,3) uint8
    if tiles is None or len(tiles) == 0:
        return {"organ": None}
    dev = _STATE["device"]
    enc = _STATE["encoder"]
    enc2 = _STATE["encoder2"]
    model = _STATE["model"]
    labels = _STATE["labels"]
    use_amp = dev == "cuda"

    def _embed(encoder):
        """Per-tile features, batched + autocast fp16 to match the offline extraction that
        produced the embeddings the head was trained on. Returns (N, D) float32."""
        out = []
        for i in range(0, x.shape[0], 128):
            xb = x[i:i + 128].to(dev, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    f = encoder(xb)
            else:
                f = encoder(xb)
            out.append(f.float().cpu())
        return torch.cat(out)

    with torch.no_grad():
        x = torch.from_numpy(tiles).permute(0, 3, 1, 2).float().div_(255.0)
        feats = _embed(enc)                             # (N, 512) CONCH, CLIP norm internal
        if _STATE["fusion"]:
            feats2 = _embed(enc2)                       # (N, 1536) UNI2-h, ImageNet norm
            feats = torch.cat([feats, feats2], dim=1)   # (N, 2048) — order matches training
        feats = feats.to(dev)
        logits = model(feats.unsqueeze(0))              # {"organ": (1,10), "dx": (1,77)}
        po_t = logits["organ"].argmax(-1)
        dx_masked = model.masked_dx_logits(logits["dx"], po_t)   # hierarchical: organ-conditioned
        conf = float(torch.softmax(dx_masked, -1).max())
        po, pd = int(po_t.item()), int(dx_masked.argmax(-1).item())

    organ_list, dx_list = labels["organ"], labels["dx"]
    organ_str = organ_list[po] if 0 <= po < len(organ_list) else None
    _, dx_str = dx_label_to_organ_dx(dx_list[pd]) if 0 <= pd < len(dx_list) else (None, None)
    # prostate tumor-detection specialist override: the main head's single most
    # concentrated error is the prostate No-tumor<->Acinar adenocarcinoma boundary.
    # When a specialist is bundled and the main head lands on one of those two, let
    # the focused binary detector decide. Fully guarded -> any failure is a no-op.
    _PROSTATE_BIN = ("No tumor present", "Acinar adenocarcinoma")
    if _STATE.get("spec") is not None and organ_str == "Prostate" and dx_str in _PROSTATE_BIN:
        try:
            with torch.no_grad():
                tumor = int(_STATE["spec"](feats.unsqueeze(0)).argmax(-1).item())
            dx_str = "Acinar adenocarcinoma" if tumor == 1 else "No tumor present"
        except Exception as e:
            print(f"[predictor] prostate specialist override skipped ({e})")
    # confidence-based abstention: low-confidence dx backs off to the organ template
    tau = float(labels.get("abstain_tau", 0.0))
    if conf < tau:
        dx_str = None
    return {"organ": organ_str, "dx": dx_str}
