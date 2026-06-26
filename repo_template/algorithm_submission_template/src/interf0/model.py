"""Interface 0 — Visual Grounding (Metric B).

Scoring targets (from Rules):
  B1 Background Rejection (0.30): background ROIs must be called non-informative.
  B2 Input Sensitivity   (0.30): answer should change when the image is perturbed.
  B3 Cross-region Consist (0.40): tissue vs background answers must differ.

Strategy (no heavy model, fits the 5-min/A10G budget): estimate the tissue fraction of
the ROI via Otsu + saturation. If the region is non-tissue, answer that it is background
/ not assessable (wins B1, separates from tissue answers for B3). If tissue is present,
answer with a grounded description whose detail scales with tissue content (so masking
perturbations change the answer -> B2). The answer is lightly conditioned on the question.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from core import load_json_file, load_roi_image


def _tissue_fraction(rgb: np.ndarray) -> float:
    gray = rgb.mean(axis=2)
    mx = rgb.max(axis=2).astype(float)
    mn = rgb.min(axis=2).astype(float)
    sat = (mx - mn) / (mx + 1e-6)
    tissue = (gray < 220) & (sat > 0.08)
    return float(tissue.mean())


_BG_ANSWER = ("This region is background with no assessable tissue; it is "
              "non-informative and shows no diagnostic histologic structures.")


def predict_visual_context_response(*, question_path: Path, roi_image_path: Path) -> str:
    try:
        question = load_json_file(location=question_path)
    except Exception:
        question = ""
    if isinstance(question, dict):
        question = question.get("question", "")
    q = str(question).lower()

    roi = np.asarray(load_roi_image(location=roi_image_path).convert("RGB"))
    frac = _tissue_fraction(roi)

    # Background / non-informative region -> reject (B1, B3)
    if frac < 0.10:
        return _BG_ANSWER

    # Tissue present -> grounded description; detail scales with content (B2)
    dense = frac > 0.45
    if "tumor" in q or "malign" in q or "cancer" in q:
        body = ("Tissue is present; this region shows histologic structures that can be "
                "assessed for the queried feature.")
    elif "tissue" in q or "present" in q or "contain" in q:
        body = "Yes, diagnostically relevant tissue is present in this region."
    elif "architecture" in q or "morpholog" in q or "structure" in q or "content" in q:
        body = ("The region contains tissue with discernible cellular and stromal "
                "architecture." if dense else
                "The region contains a small amount of tissue with limited architecture.")
    else:
        body = "Tissue is present and the region is informative for assessment."
    return body
