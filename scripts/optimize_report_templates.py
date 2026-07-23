#!/usr/bin/env python3
"""Optimize the per-(organ,dx) report text for the Final Report sub-metric.

The deterministic template currently emits the report of the MODAL-edge-set representative
case for each (organ,dx) key -- chosen for graph purity, not report quality. Here we instead
pick, per key, the **medoid report**: the candidate training report that maximizes the mean
report sub-score against that key's training-report distribution. This targets the
highest-weight component (Final Report = 0.40 within workflow A) without any model change.

We measure the held-out gain with TRUE (organ,dx) so it isolates the template effect from
diagnosis errors (comparable to the oracle organ+dx report = 0.805 ceiling row).

Outputs (when --write): artifacts/report_medoids.json  { "<organ>||<dx>": "<best report text>" }
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np
from collections import defaultdict, Counter

from reg2026.canon import canon
from reg2026.templates import build_templates
from reg2026 import report_metric as RM

ROOT = "/group/anantm-g00/REG2026"
DATA = f"{ROOT}/Data/train_CoT.json"
REPORT_Q = "what is the final pathology report"
ORGAN_Q = "what is the organ"
DX_Q = "what is the #1 diagnosis"
MIN_SUPPORT = 15            # only optimize keys with enough cases; smaller keep modal
W_T, W_K, W_E = RM.REG25_TEXT_WEIGHT, RM.REG25_KEY_WEIGHT, RM.REG25_EMB_WEIGHT


def field(cot, qcanon):
    for st in cot:
        if canon(st.get("question")) == qcanon:
            return (st.get("answer") or "").strip()
    return None


def hash_split(cases, frac=0.8, seed=0):
    import hashlib
    tr, va = [], []
    for c in cases:
        h = (int(hashlib.md5(f"{seed}:{c['id']}".encode()).hexdigest()[:8], 16) & 0xFFFFFFFF) / 0xFFFFFFFF
        (tr if h < frac else va).append(c)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write artifacts/report_medoids.json")
    args = ap.parse_args()

    data = json.load(open(DATA))
    tr, va = hash_split(data)
    print(f"train={len(tr)} val={len(va)}", flush=True)

    # ---- gather reports per (organ,dx) key on the TRAIN split only ----
    key_reports = defaultdict(Counter)          # key -> Counter(report_text -> n)
    for c in tr:
        cot = c.get("chain-of-thought", [])
        if not cot:
            continue
        o, d = field(cot, ORGAN_Q) or "?", field(cot, DX_Q) or "?"
        rep = field(cot, REPORT_Q)
        if rep:
            key_reports[f"{o}||{d}"][rep] += 1

    # ---- cache keyword sets + embeddings for every unique report text ----
    rm = RM.ReportMetric(with_models=True)      # scispaCy + PubMedBERT
    uniq = sorted({t for ctr in key_reports.values() for t in ctr})
    print(f"unique train reports: {len(uniq)} -- embedding + NER (cached once)...", flush=True)
    t0 = time.time()
    kw = {}; emb = {}
    for i, t in enumerate(uniq):
        kw[t] = set(rm.kw.keywords(t))
        emb[t] = rm.emb._embed(t)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(uniq)}  ({time.time()-t0:.0f}s)", flush=True)
    # normalize embeddings for cosine
    for t in uniq:
        v = emb[t]; n = np.linalg.norm(v) + 1e-8; emb[t] = v / n

    def jacc(a, b):
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    def emb_cos(a, b):                          # rescaled (s-0.5)/0.5 clamp, like the metric
        s = float(np.dot(emb[a], emb[b]))
        if s > 0.5:
            s = (s - 0.5) / 0.5
        return max(0.0, min(1.0, s))

    # ---- pick medoid report per key (unique x unique, weighted by count) ----
    medoid = {}
    for key, ctr in key_reports.items():
        texts = list(ctr); counts = np.array([ctr[t] for t in texts], float)
        total = counts.sum()
        if total < MIN_SUPPORT or len(texts) == 1:
            medoid[key] = ctr.most_common(1)[0][0]    # keep modal report for thin keys
            continue
        best_t, best_s = None, -1
        for cand in texts:
            # mean report-score of `cand` vs the key's report distribution
            s = 0.0
            for j, m in enumerate(texts):
                bleu = RM.bleu4(m, cand); rouge = RM.rouge_l(m, cand)
                sc = W_T * (bleu + rouge) + W_K * jacc(kw[cand], kw[m]) + W_E * emb_cos(cand, m)
                s += counts[j] * min(1.0, sc)
            s /= total
            if s > best_s:
                best_s, best_t = s, cand
        medoid[key] = best_t

    # ---- measure held-out report sub-score: modal vs medoid, using TRUE (organ,dx) ----
    tpl = build_templates(tr)
    modal_report = {}                           # key -> report in the modal template
    for key, cot in tpl["by_organ_dx"].items():
        modal_report[key] = field(cot, REPORT_Q) or ""

    n = 0; s_modal = s_medoid = 0.0
    miss = 0
    for c in va:
        cot = c.get("chain-of-thought", [])
        gt_rep = field(cot, REPORT_Q)
        if not gt_rep:
            continue
        o, d = field(cot, ORGAN_Q) or "?", field(cot, DX_Q) or "?"
        key = f"{o}||{d}"
        mr = modal_report.get(key)
        if mr is None:                          # key unseen in train -> both fall back equally
            miss += 1; continue
        pr = medoid.get(key, mr)
        s_modal += rm.evaluate(gt_rep, mr)["final_report_score"]
        s_medoid += rm.evaluate(gt_rep, pr)["final_report_score"]
        n += 1
        if n % 400 == 0:
            print(f"  eval {n}  modal={s_modal/n:.4f} medoid={s_medoid/n:.4f}", flush=True)

    print(f"\n=== held-out report sub-score (TRUE organ,dx; n={n}, {miss} unseen-key skipped) ===")
    print(f"  modal template report : {s_modal/n:.4f}   (oracle organ+dx row = 0.805)")
    print(f"  medoid-optimized report: {s_medoid/n:.4f}")
    print(f"  delta                 : {(s_medoid-s_modal)/n:+.4f}")
    print(f"  -> overall-score effect if dx correct ~74%: "
          f"{(s_medoid-s_modal)/n * 0.40 * 0.70 * 0.74:+.4f}")

    if args.write:
        out = f"{ROOT}/artifacts/report_medoids.json"
        json.dump(medoid, open(out, "w"))
        print(f"\nwrote {out} ({len(medoid)} keys)")


if __name__ == "__main__":
    main()
