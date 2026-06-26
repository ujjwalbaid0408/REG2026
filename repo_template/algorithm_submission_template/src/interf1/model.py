"""Interface 1 — Workflow Reasoning.

Reads a WSI, predicts (organ, #1 diagnosis [+ gradings]) via the pluggable predictor,
and emits the modal training chain-of-thought for that key (deterministic template).
Falls back gracefully: dx-template -> organ-template -> global modal.

Output: bare JSON array of {question, answer, next_question}; last next_question == "".
"""
from __future__ import annotations
from pathlib import Path

from core import MODEL_PATH
from src.common.templates import load_templates, apply_template
from src.common.predictor import predict_fields

_TEMPLATES = None


def _templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        for cand in (MODEL_PATH / "templates_full.json",
                     Path(__file__).resolve().parents[1] / "common" / "templates_full.json"):
            if cand.exists():
                _TEMPLATES = load_templates(cand)
                break
        if _TEMPLATES is None:
            raise FileNotFoundError("templates_full.json not found in MODEL_PATH or src/common")
    return _TEMPLATES


def predict_chain_of_thought(*, wsi_path: Path):
    tpl = _templates()
    try:
        pred = predict_fields(wsi_path, MODEL_PATH, tpl)
    except Exception as e:
        print(f"[interf1] prediction failed ({e}); falling back to global template")
        pred = {"organ": None}
    cot = apply_template(pred, tpl)
    if not cot:
        cot = apply_template({"organ": None}, tpl)
    return cot
