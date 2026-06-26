"""Offline proxies for the REG2026 Workflow Reasoning score.

The official metric (from the Rules page) per case:
    A = 0.05*BPV + 0.30*Edge-F1 + 0.25*MESS + 0.40*FinalReport

We can compute BPV and Edge-F1 exactly. MESS uses OpenBioLLM-Llama3-70B embeddings
(too heavy for local dev), and FinalReport uses PubMedBERT embeddings + biomedical
keyword extraction. For local iteration we use transparent PROXIES:
  - MESS_proxy: token-set Jaccard between predicted and GT answers on shared GT edges
                (missing edge -> 0). Correlates with semantic similarity for these
                short, templated clinical answers.
  - Report_proxy: 0.15*ROUGE-L_f1 + 0.40*keyword_Jaccard + 0.30*token_cosine
                (keyword set = lowercased alpha tokens minus stopwords).
These are LOWER-BOUND-ish stand-ins; absolute numbers differ from the platform but
ranking of our own variants is what we use them for.
"""
import re
from .canon import canon, edge_set

_STOP = set("a an the is are was were of in on at to and or with without no not yes "
            "there any present this that for as it be more than has have show shows "
            "showing seen which what where its their".split())


def _tok(s):
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower())]


def _kw(s):
    return {t for t in _tok(s) if t not in _STOP and len(t) > 2}


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    return len(A & B) / len(A | B)


def _rouge_l_f1(pred, gt):
    p, g = _tok(pred), _tok(gt)
    if not p or not g:
        return 0.0
    # LCS
    n, m = len(p), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == g[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / n, lcs / m
    return 2 * prec * rec / (prec + rec)


def bpv(pred_cot, gt_cot):
    return 1.0 if edge_set(pred_cot) == edge_set(gt_cot) else 0.0


def edge_f1(pred_cot, gt_cot):
    P, G = edge_set(pred_cot), edge_set(gt_cot)
    if not P and not G:
        return 1.0
    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def _answers_by_edge(cot):
    """Map canonical edge -> answer string (answer attached to the source step)."""
    d = {}
    for st in cot:
        q = canon(st.get("question"))
        nq = canon(st.get("next_question"))
        if nq:
            d[(q, nq)] = st.get("answer", "")
    return d


_FINAL_Q = "what is the final pathology report"


def _is_final_edge(e):
    return e[0] == _FINAL_Q or e[1] == _FINAL_Q or "final pathology report" in e[1]


def mess_proxy(pred_cot, gt_cot):
    """Avg token-Jaccard of answers over NON-FINAL GT edges; missing edge -> 0."""
    P = _answers_by_edge(pred_cot)
    G = _answers_by_edge(gt_cot)
    nonfinal = [e for e in G if not _is_final_edge(e)]
    if not nonfinal:
        return 1.0
    tot = 0.0
    for e in nonfinal:
        if e in P:
            tot += _jaccard(_tok(P[e]), _tok(G[e]))
        # else 0
    return tot / len(nonfinal)


def _final_report_text(cot):
    for st in cot:
        if canon(st.get("question")) == _FINAL_Q:
            return st.get("answer", "")
    # fallback: the answer on the step whose next_question is final report? No—report is the answer to the final Q.
    return ""


def report_proxy(pred_cot, gt_cot):
    pred = _final_report_text(pred_cot)
    gt = _final_report_text(gt_cot)
    if not gt:
        return 1.0 if not pred else 0.0
    rouge = _rouge_l_f1(pred, gt)
    kw = _jaccard(_kw(pred), _kw(gt))
    # token cosine via multiset overlap proxy
    pc, gc = _kw(pred), _kw(gt)
    cos = len(pc & gc) / ((len(pc) ** 0.5) * (len(gc) ** 0.5)) if pc and gc else 0.0
    return max(0.0, min(1.0, 0.15 * rouge + 0.40 * kw + 0.30 * cos))


def workflow_score(pred_cot, gt_cot):
    b = bpv(pred_cot, gt_cot)
    ef = edge_f1(pred_cot, gt_cot)
    me = mess_proxy(pred_cot, gt_cot)
    rp = report_proxy(pred_cot, gt_cot)
    total = 0.05 * b + 0.30 * ef + 0.25 * me + 0.40 * rp
    return dict(total=total, bpv=b, edge_f1=ef, mess=me, report=rp)
