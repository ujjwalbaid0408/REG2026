"""Canonicalization helpers shared by templates, metrics, and inference.

The challenge metric canonicalizes question / next_question strings by normalizing
capitalization, whitespace, and trailing punctuation before comparing edges. We must
emit the EXACT canonical training strings, so we keep a raw->canonical map and always
emit the most common raw spelling for a given canonical form.
"""
import re

def canon(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" .?!,;:")
    return s


def edge_set(cot):
    """Return the canonical edge set {(q, nq)} for a chain-of-thought list."""
    edges = set()
    for st in cot:
        q = canon(st.get("question"))
        nq = canon(st.get("next_question"))
        if nq:
            edges.add((q, nq))
    return edges
