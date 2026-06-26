"""Supervised target extraction for the REG2026 classifier.

Per oracle analysis, predicting (organ, #1 diagnosis) drives the workflow score to ~0.89,
and #1 diagnosis already encodes most grading (e.g. "Adenocarcinoma, moderately
differentiated"). So the classifier targets are: organ (10-way) and dx (134-way).
Rare dx classes (< MIN_DX) are folded to an "<organ>:other" bucket so the head stays
learnable and inference falls back to the organ template for them.
"""
import json
from collections import Counter
from .canon import canon

ORGAN_Q = "what is the organ"
DX_Q = "what is the #1 diagnosis"
MIN_DX = 15  # diagnoses rarer than this are bucketed as "<organ>:other"


def field(cot, qcanon):
    for st in cot:
        if canon(st.get("question")) == qcanon:
            return (st.get("answer") or "").strip()
    return None


def build_label_space(train_cot, min_dx=MIN_DX):
    organs = Counter()
    dxs = Counter()
    rows = {}
    for case in train_cot:
        cot = case.get("chain-of-thought", [])
        o = field(cot, ORGAN_Q) or "Unknown"
        d = field(cot, DX_Q) or "Unknown"
        organs[o] += 1
        dxs[(o, d)] += 1
        rows[case["id"]] = (o, d)
    keep_dx = {od for od, c in dxs.items() if c >= min_dx}

    def dx_label(o, d):
        return f"{o}||{d}" if (o, d) in keep_dx else f"{o}||other"

    organ_list = sorted(organs)
    dx_list = sorted({dx_label(o, d) for (o, d) in dxs})
    organ_idx = {o: i for i, o in enumerate(organ_list)}
    dx_idx = {d: i for i, d in enumerate(dx_list)}

    labels = {}
    for cid, (o, d) in rows.items():
        labels[cid] = {"organ": organ_idx[o], "dx": dx_idx[dx_label(o, d)],
                       "organ_str": o, "dx_str": d}
    return dict(organ_list=organ_list, dx_list=dx_list, labels=labels,
               min_dx=min_dx)


def dx_label_to_organ_dx(dx_label):
    """Map a dx-head label string '<organ>||<diagnosis>' back to (organ, dx or None)."""
    organ, _, dx = dx_label.partition("||")
    return organ, (None if dx == "other" else dx)
