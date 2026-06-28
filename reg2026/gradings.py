"""Grading sub-head label space for REG2026.

Many diagnoses are organ + grade (e.g. prostate adenocarcinoma is defined by its Gleason
score / grade group; breast IC-NST by its Nottingham grade). The reasoning graph asks these
as explicit categorical questions. We turn each into an auxiliary classification head on the
shared MIL trunk:

  * auxiliary supervision sharpens the shared features -> better (organ, dx) accuracy;
  * several of these answers are also template SUBSTITUTABLE fields
    (gleason_score, grade_group, nott_overall, grade), so predicting them correctly
    *directly* improves Edge-F1 / MESS on those Q&A nodes.

Each field is sparse (only present for the relevant organ), so its loss is masked per-case
(class index -1 = absent -> CrossEntropyLoss ignore_index).
"""
from __future__ import annotations
import collections

# canonical question (lower, no trailing '?') -> field key
GRADING_QUESTIONS = {
    "what is the grade of neoplasm": "grade",
    "what is the gleason score": "gleason_score",
    "what is the grade group": "grade_group",
    "what is the worst grade pattern": "worst_gleason",
    "what is the score for tubular differentiation": "nott_tubule",
    "what is the score for nuclear pleomorphism": "nott_nuclear",
    "what is the score for mitotic rate": "nott_mitotic",
    "what is the overall score": "nott_overall",
    "what is the grade of dysplasia": "dysplasia",
    "what is the nuclear grade of lesion": "nuclear_grade",
}

# Stable field order for head indexing.
GRADING_FIELDS = list(dict.fromkeys(GRADING_QUESTIONS.values()))


def _canon(q: str) -> str:
    return " ".join(str(q).strip().lower().split()).rstrip("?")


def build_grading_space(data, min_count: int = 5):
    """Returns:
        vocabs:  {field: {answer_str: class_idx}}  (answers seen >= min_count)
        labels:  {case_id: {field: class_idx}}     (only fields present & in-vocab)
        dims:    [(field, n_classes), ...]          in GRADING_FIELDS order
    """
    counters = collections.defaultdict(collections.Counter)
    raw = collections.defaultdict(dict)   # cid -> {field: answer_str}
    for r in data:
        cid = r["id"]
        for s in r.get("chain-of-thought", []):
            q = _canon(s.get("question", ""))
            f = GRADING_QUESTIONS.get(q)
            if f is not None:
                ans = str(s.get("answer", "")).strip()
                if ans:
                    counters[f][ans] += 1
                    raw[cid][f] = ans

    vocabs = {}
    for f in GRADING_FIELDS:
        keep = [a for a, c in counters[f].most_common() if c >= min_count]
        vocabs[f] = {a: i for i, a in enumerate(keep)}

    labels = {}
    for cid, fa in raw.items():
        d = {}
        for f, ans in fa.items():
            idx = vocabs[f].get(ans)
            if idx is not None:
                d[f] = idx
        if d:
            labels[cid] = d

    dims = [(f, len(vocabs[f])) for f in GRADING_FIELDS]
    return {"vocabs": vocabs, "labels": labels, "dims": dims}


def target_vector(case_labels, dims):
    """case_labels: {field: class_idx} (may be partial). Returns a list aligned to `dims`
    with -1 for absent fields (CrossEntropyLoss ignore_index)."""
    cl = case_labels or {}
    return [cl.get(f, -1) for f, _ in dims]
