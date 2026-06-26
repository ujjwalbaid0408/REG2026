#!/usr/bin/env python3
"""Quantify the templating ceiling on a held-out split.

Builds templates on 80% of train, evaluates the Workflow Reasoning score (proxy) on the
held-out 20% under several oracle conditions, to answer:
  - How much score does organ-only prediction give?  (floor for a weak classifier)
  - How much does adding #1-diagnosis give?           (the main classifier target)
  - How much headroom remains in report generation?   (oracle report upper bound)
"""
import json, sys, os
sys.path.insert(0, "/group/anantm-g00/REG2026")
from collections import defaultdict
from reg2026.canon import canon
from reg2026.templates import build_templates, apply_template, ORGAN_Q, DX_Q, _field
from reg2026 import metrics

DATA = "/group/anantm-g00/REG2026/Data/train_CoT.json"
OUT = "/group/anantm-g00/REG2026/artifacts"
os.makedirs(OUT, exist_ok=True)


def split(data, frac=0.8, seed=0):
    # deterministic hash split by id (stable across processes; matches train_mil.hash_split)
    import hashlib
    tr, va = [], []
    for c in data:
        digest = hashlib.md5(f"{seed}:{c['id']}".encode()).hexdigest()
        h = (int(digest[:8], 16) & 0xFFFFFFFF) / 0xFFFFFFFF
        (tr if h < frac else va).append(c)
    return tr, va


def report_text(cot):
    return metrics._final_report_text(cot)


def mean(d):
    return {k: round(sum(x[k] for x in d) / len(d), 4) for k in d[0]}


def main():
    data = json.load(open(DATA))
    tr, va = split(data)
    print(f"train {len(tr)}  val {len(va)}")
    tpl = build_templates(tr)
    json.dump(tpl, open(f"{OUT}/templates_trainsplit.json", "w"))

    conditions = {
        "global_only": lambda gt: dict(organ=None),
        "oracle_organ": lambda gt: dict(organ=_field(gt, ORGAN_Q)),
        "oracle_organ_dx": lambda gt: dict(organ=_field(gt, ORGAN_Q), dx=_field(gt, DX_Q)),
        "oracle_organ_dx_report": lambda gt: dict(
            organ=_field(gt, ORGAN_Q), dx=_field(gt, DX_Q), report=report_text(gt)),
    }
    results = {}
    per_organ = defaultdict(lambda: defaultdict(list))
    for name, predfn in conditions.items():
        scores = []
        for c in va:
            gt = c["chain-of-thought"]
            pred = predfn(gt)
            cot = apply_template(pred, tpl)
            s = metrics.workflow_score(cot, gt)
            scores.append(s)
            per_organ[name][_field(gt, ORGAN_Q) or "?"].append(s["total"])
        results[name] = mean(scores)
        print(f"\n== {name} ==")
        for k, v in results[name].items():
            print(f"   {k:10s} {v}")

    print("\n== oracle_organ_dx by organ (total) ==")
    for organ, lst in sorted(per_organ["oracle_organ_dx"].items(),
                             key=lambda kv: -len(kv[1])):
        print(f"   {organ:18s} n={len(lst):5d}  total={round(sum(lst)/len(lst),4)}")

    json.dump(results, open(f"{OUT}/oracle_eval.json", "w"), indent=2)
    print(f"\n[written] {OUT}/oracle_eval.json")


if __name__ == "__main__":
    main()
