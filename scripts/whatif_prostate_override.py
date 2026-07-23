#!/usr/bin/env python3
"""Measure the REAL workflow gain from the prostate specialist as an override.

Runs the main fusion model (hierarchical) on held-out, then for slides where the
main model predicts Prostate + {No tumor, Acinar adenocarcinoma}, replaces that
binary call with the specialist's decision. Recomputes mean workflow score vs the
unmodified baseline. This is the achievable (not oracle) delta.
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np, torch
from torch.utils.data import DataLoader

from scripts.train_mil import hash_split, SlideDS, collate
import scripts.train_fusion_mil as F
from scripts.train_prostate_specialist import ProstateSpecialist
from reg2026.labels import build_label_space, dx_label_to_organ_dx, field, ORGAN_Q
from reg2026.templates import build_templates, apply_template
from reg2026.mil import MILClassifier
from reg2026 import metrics

MAIN = sys.argv[1] if len(sys.argv) > 1 else "f2_fuse_dxw"
SPEC = sys.argv[2] if len(sys.argv) > 2 else "prostate_specialist"
NO_TUMOR, ACINAR = "No tumor present", "Acinar adenocarcinoma"

ck = torch.load(f"{F.OUT_ROOT}/{MAIN}/mil_head.pt", map_location="cpu", weights_only=False)
model = MILClassifier.from_config(ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()
sck = torch.load(f"{F.OUT_ROOT}/{SPEC}/spec_head.pt", map_location="cpu", weights_only=False)
spec = ProstateSpecialist(in_dim=sck["in_dim"]); spec.load_state_dict(sck["state_dict"]); spec.eval()
print(f"# specialist held-out: acc={sck['best']['acc']:.4f} bal_acc={sck['best']['bacc']:.4f}")

data = json.load(open(F.DATA))
tr, va = hash_split(data)
ls = build_label_space(data)
organ_list, dx_list, labels = ls["organ_list"], ls["dx_list"], ls["labels"]
tpl = build_templates(tr)
cache, _, _ = F.load_fusion_cache([c["id"] for c in va])
cases_by_id = {c["id"]: c for c in va}

va_ld = DataLoader(SlideDS(va, labels, cache, train=False), batch_size=1,
                   shuffle=False, collate_fn=collate, num_workers=2)
items = va_ld.dataset.items

base_tot = ovr_tot = 0.0
n = 0; n_override = 0; n_flip = 0; flip_correct = 0
idx = 0
for x, mask, o, d in va_ld:
    with torch.no_grad():
        out = model(x, mask)
    po = int(out["organ"].argmax(-1)[0])
    pd = int(model.masked_dx_logits(out["dx"], out["organ"].argmax(-1)).argmax(-1)[0])
    cid = items[idx][0]; idx += 1
    gt = cases_by_id[cid]["chain-of-thought"]
    p_organ = organ_list[po]; _, p_dx = dx_label_to_organ_dx(dx_list[pd])
    gt_organ = field(gt, ORGAN_Q) or "?"; _, gt_dx = dx_label_to_organ_dx(dx_list[int(d[0])])

    base_dx = p_dx
    ovr_dx = p_dx
    if p_organ == "Prostate" and p_dx in (NO_TUMOR, ACINAR):
        n_override += 1
        with torch.no_grad():
            tumor = int(spec(x, mask).argmax(-1)[0])
        new_dx = ACINAR if tumor == 1 else NO_TUMOR
        if new_dx != p_dx:
            n_flip += 1
            flip_correct += int(new_dx == gt_dx) - int(p_dx == gt_dx)
        ovr_dx = new_dx

    s_base = metrics.workflow_score(apply_template({"organ": p_organ, "dx": base_dx}, tpl), gt)
    s_ovr = metrics.workflow_score(apply_template({"organ": p_organ, "dx": ovr_dx}, tpl), gt)
    base_tot += s_base["total"]; ovr_tot += s_ovr["total"]; n += 1

print(f"# n={n}  prostate-binary predictions overridden={n_override}  flips={n_flip}  "
      f"net_correct_flips={flip_correct:+d}")
print(f"# BASELINE workflow = {base_tot/n:.4f}")
print(f"# +SPECIALIST       = {ovr_tot/n:.4f}   (Δ {ovr_tot/n - base_tot/n:+.4f})")
