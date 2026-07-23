#!/usr/bin/env python3
"""What-if ROI: size each dx-fix bucket in WORKFLOW-SCORE terms (not just dx acc).

For the best fusion model, run hierarchical inference -> predicted (organ,dx) per
held-out slide, score the emitted template CoT vs GT (baseline). Then for each
candidate "fix" bucket, replace the prediction with the ORACLE label for slides in
that bucket and recompute the mean workflow score. The delta = the maximum
achievable workflow gain if that bucket were solved perfectly. This decides where
(if anywhere) a specialist head is worth building.
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np, torch
from torch.utils.data import DataLoader

import scripts.train_mil as T
from scripts.train_mil import hash_split, SlideDS, collate
import scripts.train_fusion_mil as F
from reg2026.labels import build_label_space, dx_label_to_organ_dx, field, ORGAN_Q
from reg2026.templates import build_templates, apply_template
from reg2026.mil import MILClassifier
from reg2026 import metrics

NAME = sys.argv[1] if len(sys.argv) > 1 else "f2_fuse_dxw"

out_dir = os.path.join(F.OUT_ROOT, NAME)
ck = torch.load(os.path.join(out_dir, "mil_head.pt"), map_location="cpu", weights_only=False)
model = MILClassifier.from_config(ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()

data = json.load(open(F.DATA))
tr, va = hash_split(data)
ls = build_label_space(data)
organ_list, dx_list, labels = ls["organ_list"], ls["dx_list"], ls["labels"]
tpl = build_templates(tr)
cache, miss, mism = F.load_fusion_cache([c["id"] for c in va])
cases_by_id = {c["id"]: c for c in va}

va_ld = DataLoader(SlideDS(va, labels, cache, train=False), batch_size=32,
                   shuffle=False, collate_fn=collate, num_workers=4)
items = va_ld.dataset.items

# ---- collect per-slide predictions ----
recs = []  # dict per slide: id, gt_organ, gt_dx_name, pred_organ, pred_dx_name, true_o,true_d idx
idx = 0
for x, mask, o, d in va_ld:
    with torch.no_grad():
        out = model(x, mask)
    po_t = out["organ"].argmax(-1)
    pd_t = model.masked_dx_logits(out["dx"], po_t).argmax(-1)
    for k in range(len(o)):
        cid = items[idx][0]; idx += 1
        gt = cases_by_id[cid]["chain-of-thought"]
        gt_organ = field(gt, ORGAN_Q) or "?"
        _, gt_dx = dx_label_to_organ_dx(dx_list[int(d[k])])
        p_organ = organ_list[int(po_t[k])]
        _, p_dx = dx_label_to_organ_dx(dx_list[int(pd_t[k])])
        recs.append(dict(id=cid, gt=gt, gt_organ=gt_organ, gt_dx=gt_dx,
                         true_o=int(o[k]), true_d=int(d[k]),
                         p_organ=p_organ, p_dx=p_dx))

def score_all(pick):
    """pick(rec) -> (organ_str, dx_str) to emit. returns mean workflow + components."""
    tot = defaultdict(float)
    for r in recs:
        org, dx = pick(r)
        cot = apply_template({"organ": org, "dx": dx}, tpl)
        s = metrics.workflow_score(cot, r["gt"])
        for kk in ("total", "edge_f1", "mess", "report", "bpv"):
            tot[kk] += s[kk]
    n = len(recs)
    return {kk: tot[kk] / n for kk in tot}

base_pick   = lambda r: (r["p_organ"], r["p_dx"])
oracle_pick = lambda r: (r["gt_organ"], r["gt_dx"])

base = score_all(base_pick)
orac = score_all(oracle_pick)
print(f"# n={len(recs)} held-out slides, model={NAME}")
print(f"{'scenario':<42}{'workflow':>9}{'Δ':>8}{'edgeF1':>8}{'mess':>7}{'report':>8}")
def show(tag, s, b=base):
    print(f"{tag:<42}{s['total']:>9.4f}{s['total']-b['total']:>+8.4f}"
          f"{s['edge_f1']:>8.3f}{s['mess']:>7.3f}{s['report']:>8.3f}")
show("BASELINE (predicted organ+dx)", base, base)
show("ORACLE organ+dx (ceiling)", orac)

# ---- bucket fixes: oracle ONLY for slides in the bucket ----
def fix_if(cond):
    return lambda r: (r["gt_organ"], r["gt_dx"]) if cond(r) else (r["p_organ"], r["p_dx"])

print("\n# --- per-organ: oracle this organ only ---")
organs = sorted({r["gt_organ"] for r in recs})
rows = []
for org in organs:
    n = sum(r["gt_organ"] == org for r in recs)
    s = score_all(fix_if(lambda r, org=org: r["gt_organ"] == org))
    rows.append((org, n, s["total"] - base["total"]))
for org, n, dlt in sorted(rows, key=lambda x: -x[2]):
    print(f"  {org:<18} n={n:<5} Δworkflow={dlt:+.4f}")

print("\n# --- structural buckets ---")
# 1. same disease-name, wrong organ index (rectum/colon aliasing)
samename = fix_if(lambda r: r["p_dx"] == r["gt_dx"] and r["p_organ"] != r["gt_organ"])
n_sn = sum(r["p_dx"] == r["gt_dx"] and r["p_organ"] != r["gt_organ"] for r in recs)
show(f"fix same-name/wrong-organ (n={n_sn})", score_all(samename))

# 2. prostate tumor vs no-tumor (the No tumor <-> Acinar adenocarcinoma pair)
PT = {"No tumor present", "Acinar adenocarcinoma"}
prost_bin = fix_if(lambda r: r["gt_organ"] == "Prostate" and r["gt_dx"] in PT and r["p_dx"] in PT and r["p_dx"] != r["gt_dx"])
n_pb = sum(r["gt_organ"]=="Prostate" and r["gt_dx"] in PT and r["p_dx"] in PT and r["p_dx"]!=r["gt_dx"] for r in recs)
show(f"fix prostate tumor/no-tumor (n={n_pb})", score_all(prost_bin))

# 3. organ-only oracle (keep predicted dx, fix organ) -> isolates organ-routing value
org_only = lambda r: (r["gt_organ"], r["p_dx"])
show("fix ORGAN only (keep pred dx)", score_all(org_only))

# 4. dx-only oracle (keep predicted organ, fix dx)
dx_only = lambda r: (r["p_organ"], r["gt_dx"])
show("fix DX only (keep pred organ)", score_all(dx_only))
