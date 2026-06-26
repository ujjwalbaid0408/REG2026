#!/usr/bin/env python3
"""Detailed held-out evaluation of a trained MIL checkpoint for the report:
per-organ workflow score, organ/dx accuracy, and qualitative sample predictions
(predicted CoT vs ground truth). Writes artifacts/mil/<name>/eval_detail.json.
"""
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np, torch

import scripts.train_mil as T
from reg2026.labels import build_label_space, dx_label_to_organ_dx, field, ORGAN_Q
from reg2026.templates import build_templates, apply_template
from reg2026.mil import MILClassifier
from reg2026 import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="r1_reg")
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()
    out_dir = os.path.join(T.OUT_ROOT, args.name)
    ck = torch.load(os.path.join(out_dir, "mil_head.pt"), map_location="cpu")
    maps = json.load(open(os.path.join(out_dir, "label_maps.json")))
    organ_list, dx_list = maps["organ"], maps["dx"]
    model = MILClassifier.from_config(ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()

    data = json.load(open(T.DATA))
    tr, va = T.hash_split(data)
    ls = build_label_space(data); labels = ls["labels"]
    tpl = build_templates(tr)
    cache = T.load_cache([c["id"] for c in va])

    per_organ = defaultdict(list); organ_hit = defaultdict(lambda: [0, 0]); dx_hit = [0, 0]
    samples = []
    for c in va:
        cid = c["id"]
        if cid not in cache:
            continue
        feats = torch.from_numpy(cache[cid]).unsqueeze(0)
        with torch.no_grad():
            out = model(feats)
        po = int(out["organ"].argmax(-1)); pd = int(out["dx"].argmax(-1))
        organ_str = organ_list[po]; _, dx_str = dx_label_to_organ_dx(dx_list[pd])
        gt = c["chain-of-thought"]; gt_organ = field(gt, ORGAN_Q) or "?"
        cot = apply_template({"organ": organ_str, "dx": dx_str}, tpl)
        s = metrics.workflow_score(cot, gt)
        per_organ[gt_organ].append(s["total"])
        organ_hit[gt_organ][0] += int(organ_str == gt_organ); organ_hit[gt_organ][1] += 1
        lab = labels.get(cid)
        if lab:
            dx_hit[0] += int(pd == lab["dx"]); dx_hit[1] += 1
        if len(samples) < args.samples and len(cot) >= 5:
            samples.append(dict(id=cid, gt_organ=gt_organ, pred_organ=organ_str,
                                pred_dx=dx_str, workflow=round(s["total"], 3),
                                edge_f1=round(s["edge_f1"], 3), mess=round(s["mess"], 3)))

    summary = {o: dict(workflow=round(np.mean(v), 4), n=len(v),
                       organ_acc=round(organ_hit[o][0] / organ_hit[o][1], 3))
               for o, v in sorted(per_organ.items(), key=lambda kv: -len(kv[1]))}
    overall = dict(workflow=round(np.mean([x for v in per_organ.values() for x in v]), 4),
                   dx_acc=round(dx_hit[0] / max(dx_hit[1], 1), 4),
                   val_n=sum(len(v) for v in per_organ.values()))
    result = dict(name=args.name, overall=overall, per_organ=summary, samples=samples)
    json.dump(result, open(os.path.join(out_dir, "eval_detail.json"), "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
