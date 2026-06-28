"""Phase-1 report diagnostic.

Decomposes the final-report sub-metric on the held-out 20% split to separate two
sources of loss:

  (A) FORMAT loss  - GT report vs the modal template report for the case's TRUE
                     (organ, dx). Isolates how much the current modal-template
                     phrasing loses even when fields are routed perfectly.
  (B) FIELD loss   - additionally degraded when (organ, dx) are predicted wrong;
                     here we report the oracle-field ceiling, so any remaining
                     gap to the leaderboard's 0.516 is attributable to dx errors.

Run with the eval venv (spacy en_core_sci_lg + PubMedBERT available).
"""
import sys, json, hashlib, argparse, collections
sys.path.insert(0, ".")
from reg2026.report_metric import ReportMetric, bleu4, rouge_l

ORGAN_Q = "what is the organ"
DX_Q = "what is the #1 diagnosis"
REPORT_Q = "what is the final pathology report"


def canon(s):
    return " ".join(str(s).strip().lower().split()).rstrip("?")


def field(cot, q):
    for s in cot:
        if canon(s.get("question", "")) == q:
            return s.get("answer", "")
    return None


def report_of(cot):
    for s in cot:
        if canon(s.get("question", "")) == REPORT_Q:
            return s.get("answer", "")
    return ""


def hash_split(cases, frac=0.8, seed=0):
    tr, va = [], []
    for c in cases:
        h = int(hashlib.md5(f"{seed}:{c['id']}".encode()).hexdigest(), 16)
        (tr if (h % 1000) / 1000.0 < frac else va).append(c)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/train_CoT.json")
    ap.add_argument("--templates", default="artifacts/templates_full.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all holdout cases")
    ap.add_argument("--text-only", action="store_true", help="skip spacy/pubmedbert")
    ap.add_argument("--no-key", action="store_true", help="skip spacy keyword term")
    ap.add_argument("--no-emb", action="store_true", help="skip pubmedbert term")
    args = ap.parse_args()

    data = json.load(open(args.data))
    tpl = json.load(open(args.templates))
    by_od = tpl["by_organ_dx"]
    by_o = tpl["by_organ"]
    _, va = hash_split(data)
    if args.limit:
        va = va[: args.limit]

    if args.text_only:
        metric = ReportMetric(with_models=False)
    else:
        metric = ReportMetric(with_key=not args.no_key, with_emb=not args.no_emb)

    agg = collections.defaultdict(float)
    per_organ = collections.defaultdict(lambda: collections.defaultdict(float))
    per_organ_n = collections.Counter()
    n = miss = 0
    for c in va:
        cot = c.get("chain-of-thought", [])
        if not cot:
            continue
        organ = field(cot, ORGAN_Q) or "?"
        dx = field(cot, DX_Q) or "?"
        gt = report_of(cot)
        if not gt:
            continue
        # current approach, oracle fields: modal report for true (organ,dx)
        cand = by_od.get(f"{organ}||{dx}") or by_o.get(organ)
        if cand is None:
            miss += 1
            continue
        pred = report_of(cand)
        sc = metric.evaluate(gt, pred)
        n += 1
        for k, v in sc.items():
            agg[k] += v
            per_organ[organ][k] += v
        per_organ_n[organ] += 1

    print(f"\nHeld-out cases scored: {n}  (template miss: {miss})")
    print("=== FORMAT ceiling: modal template report given TRUE (organ,dx) ===")
    for k in ["final_report_score", "key", "emb", "bleu", "rouge"]:
        print(f"  {k:20s} {agg[k]/n:.4f}")
    print(f"\n  (leaderboard report w/ PREDICTED fields = 0.516; "
          f"gap below this ceiling = dx-error loss)")
    print("\n=== per-organ final_report_score (oracle fields) ===")
    for o in sorted(per_organ, key=lambda x: -per_organ_n[x]):
        nn = per_organ_n[o]
        print(f"  {o:14s} n={nn:5d}  report={per_organ[o]['final_report_score']/nn:.3f}"
              f"  key={per_organ[o]['key']/nn:.3f}  emb={per_organ[o]['emb']/nn:.3f}")


if __name__ == "__main__":
    main()
