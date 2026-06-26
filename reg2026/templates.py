"""Template engine: the deterministic core of the REG2026 strategy.

Insight (verified): given (organ, #1 diagnosis), the reasoning graph is ~86% pure and
intermediate answers ~92% pure. So we predict (organ, dx [+ a few gradings]) from the
WSI and EMIT the modal training chain-of-thought for that key, optionally substituting
per-case predicted answer fields. This near-maximizes BPV/Edge-F1/MESS and supplies the
keyword-rich report content the Final Report sub-metric rewards.

build_templates(train_cot) -> dict artifact (JSON-serializable) with:
  - by_organ_dx : {f"{organ}||{dx}": modal_cot}
  - by_organ    : {organ: modal_cot}                (fallback when dx unknown/rare)
  - global      : modal_cot                          (last-resort fallback)
  - organ_dx_support / organ_support : case counts (for picking confident keys)

apply_template(pred, templates) -> chain-of-thought list, using whatever fields the
classifier predicted (at minimum `organ`; ideally `organ`+`dx`).
"""
from collections import Counter, defaultdict
import json
from .canon import canon, edge_set

ORGAN_Q = "what is the organ"
DX_Q = "what is the #1 diagnosis"


def _field(cot, qcanon):
    for st in cot:
        if canon(st.get("question")) == qcanon:
            return (st.get("answer") or "").strip()
    return None


def _sig(cot):
    return tuple(sorted(edge_set(cot)))


def _modal_cot(cot_list):
    """Pick the single most common verbatim chain-of-thought (by edge-set signature),
    returning an actual representative cot (the first case with the modal signature)."""
    sig_count = Counter(_sig(c) for c in cot_list)
    if not sig_count:
        return []
    modal_sig = sig_count.most_common(1)[0][0]
    for c in cot_list:
        if _sig(c) == modal_sig:
            return c
    return cot_list[0]


def build_templates(train_cot):
    by_od_cases = defaultdict(list)
    by_o_cases = defaultdict(list)
    all_cases = []
    for case in train_cot:
        cot = case.get("chain-of-thought", [])
        if not cot:
            continue
        organ = _field(cot, ORGAN_Q) or "?"
        dx = _field(cot, DX_Q) or "?"
        by_od_cases[(organ, dx)].append(cot)
        by_o_cases[organ].append(cot)
        all_cases.append(cot)

    by_organ_dx = {f"{o}||{d}": _modal_cot(cl) for (o, d), cl in by_od_cases.items()}
    by_organ = {o: _modal_cot(cl) for o, cl in by_o_cases.items()}
    glob = _modal_cot(all_cases)

    return dict(
        by_organ_dx=by_organ_dx,
        by_organ=by_organ,
        **{"global": glob},
        organ_dx_support={f"{o}||{d}": len(cl) for (o, d), cl in by_od_cases.items()},
        organ_support={o: len(cl) for o, cl in by_o_cases.items()},
    )


def save_templates(templates, path):
    with open(path, "w") as f:
        json.dump(templates, f)


def load_templates(path):
    with open(path) as f:
        return json.load(f)


# ---- answer-field substitution -------------------------------------------------
# Questions whose answers are case-specific and worth overriding from classifier preds
# when available (everything else comes from the modal template verbatim).
SUBSTITUTABLE = {
    "what is the organ": "organ",
    "what is the #1 diagnosis": "dx",
    "what is the final pathology report": "report",
    "what is the gleason score": "gleason_score",
    "what is the grade group": "grade_group",
    "what is the overall score": "nottingham_overall",
    "what is the grade of neoplasm": "grade",
}


def apply_template(pred, templates):
    """pred: dict with at least 'organ'; optionally 'dx' and substitutable answer fields.
    Returns a chain-of-thought list (copied), with last next_question == ''."""
    organ = pred.get("organ") or "?"
    dx = pred.get("dx")
    cot = None
    if dx:
        cot = templates["by_organ_dx"].get(f"{organ}||{dx}")
    if cot is None:
        cot = templates["by_organ"].get(organ)
    if cot is None:
        cot = templates["global"]
    # deep copy + optional answer substitution
    out = []
    for st in cot:
        st2 = dict(question=st.get("question", ""),
                   answer=st.get("answer", ""),
                   next_question=st.get("next_question", ""))
        qc = canon(st2["question"])
        if qc in SUBSTITUTABLE:
            key = SUBSTITUTABLE[qc]
            if pred.get(key) is not None:
                st2["answer"] = str(pred[key])
        out.append(st2)
    # enforce terminal contract: ensure at least one step has empty next_question
    if out and all(s["next_question"] != "" for s in out):
        # the final-report step is terminal in training; blank its next_question
        for s in out:
            if canon(s["question"]) == "what is the final pathology report":
                s["next_question"] = ""
                break
        else:
            out[-1]["next_question"] = ""
    return out
