#!/usr/bin/env python3
"""Evaluate an ENSEMBLE of trained fusion MIL heads on the held-out 20% split.

Averages per-slide softmax probabilities across the given runs (organ + dx), applies
the same hierarchical organ-conditioned dx masking and confidence-abstention sweep as
train_mil.evaluate(), and decodes (organ, dx) -> template -> workflow score. This makes
the ensemble number directly comparable to the single-model held-out scores
(f2_fuse_dxw = 0.8143, oracle ceiling = 0.889).

Usage:  python scripts/eval_ensemble.py f0_fuse f1_fuse_big f2_fuse_dxw f3_fuse_grade
No GPU required (head-only inference over cached embeddings).
"""
import sys, json, os, time
sys.path.insert(0, "/group/anantm-g00/REG2026")
sys.path.insert(0, "/group/anantm-g00/REG2026/scripts")
import numpy as np
import torch

from reg2026.labels import build_label_space, dx_label_to_organ_dx
from reg2026.templates import build_templates, apply_template
from reg2026 import metrics
from reg2026.mil import MILClassifier
from train_mil import hash_split
from train_fusion_mil import load_fusion_cache

ROOT = "/group/anantm-g00/REG2026"
DATA = f"{ROOT}/Data/train_CoT.json"
MILD = f"{ROOT}/artifacts/mil"
TAU_GRID = [round(t, 3) for t in np.linspace(0.0, 0.95, 20)]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


@torch.no_grad()
def model_probs(model, feats):
    """Return (organ_prob[n_organ], dx_prob[n_dx]) for one slide's (N,2048) features."""
    x = torch.from_numpy(feats).unsqueeze(0)              # (1,N,2048)
    mask = torch.ones(1, feats.shape[0], dtype=torch.bool)
    out = model(x, mask)
    return (softmax(out["organ"].numpy())[0], softmax(out["dx"].numpy())[0])


def main():
    runs = sys.argv[1:] or ["f0_fuse", "f1_fuse_big", "f2_fuse_dxw", "f3_fuse_grade"]
    data = json.load(open(DATA))
    split_tr, split_va = hash_split(data)
    ls = build_label_space(data)
    organ_list, dx_list, labels = ls["organ_list"], ls["dx_list"], ls["labels"]
    organ_idx = {o: i for i, o in enumerate(organ_list)}
    dx_organ = np.array([organ_idx[d.split("||")[0]] for d in dx_list])
    tpl = build_templates(split_tr)                       # templates from train split only

    t0 = time.time()
    cache, _, _ = load_fusion_cache([c["id"] for c in split_va])   # only val slides needed
    va = [c for c in split_va if c["id"] in labels and c["id"] in cache]
    cases_by_id = {c["id"]: c for c in va}
    print(f"loaded fused cache {len(cache)}; val slides {len(va)} in {time.time()-t0:.0f}s", flush=True)

    # load models
    models = []
    for r in runs:
        ck = torch.load(f"{MILD}/{r}/mil_head.pt", map_location="cpu", weights_only=False)
        m = MILClassifier.from_config(ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
        models.append((r, m))
    print(f"ensembling {len(models)} models: {[r for r,_ in models]}", flush=True)

    # collect per-model and ensemble probabilities per val slide
    rows = []            # dict per case
    for ci, c in enumerate(va):
        cid = c["id"]; feats = cache[cid]
        gt = cases_by_id[cid]["chain-of-thought"]
        o_gt, d_gt = labels[cid]["organ"], labels[cid]["dx"]
        oprobs, dprobs = [], []
        for _, m in models:
            op, dp = model_probs(m, feats)
            oprobs.append(op); dprobs.append(dp)
        rows.append(dict(cid=cid, gt=gt, o_gt=o_gt, d_gt=d_gt,
                         oprobs=np.array(oprobs), dprobs=np.array(dprobs)))
        if (ci + 1) % 500 == 0:
            print(f"  {ci+1}/{len(va)}", flush=True)

    def score_set(weights):
        """weights: bool/array over models to include (averaged). Returns metrics dict."""
        idx = np.where(weights)[0]
        no = nd = 0
        per = []           # (conf, s_dx, s_org)
        for row in rows:
            op = row["oprobs"][idx].mean(0)
            dp = row["dprobs"][idx].mean(0)
            po = int(op.argmax())
            # hierarchical mask: restrict dx to predicted organ's classes
            md = np.where(dx_organ == po, dp, -1.0)
            pd = int(md.argmax())
            conf = float(dp[pd])
            no += (po == row["o_gt"]); nd += (pd == row["d_gt"])
            organ_str = organ_list[po]
            _, dx_str = dx_label_to_organ_dx(dx_list[pd])
            s_dx = metrics.workflow_score(apply_template({"organ": organ_str, "dx": dx_str}, tpl), row["gt"])["total"]
            s_org = metrics.workflow_score(apply_template({"organ": organ_str, "dx": None}, tpl), row["gt"])["total"]
            per.append((conf, s_dx, s_org))
        n = len(rows)
        wf = float(np.mean([p[1] for p in per]))
        best_wf, best_tau = wf, 0.0
        for tau in TAU_GRID:
            mwf = float(np.mean([(p[1] if p[0] >= tau else p[2]) for p in per]))
            if mwf > best_wf:
                best_wf, best_tau = mwf, tau
        return dict(organ_acc=no/n, dx_acc=nd/n, workflow=wf, workflow_abstain=best_wf, tau=best_tau)

    print("\n=== single models (recomputed under this protocol) ===", flush=True)
    for i, (r, _) in enumerate(models):
        w = np.zeros(len(models), bool); w[i] = True
        s = score_set(w)
        print(f"  {r:16s} organ={s['organ_acc']:.3f} dx={s['dx_acc']:.3f} "
              f"wf={s['workflow']:.4f} wf_abstain={s['workflow_abstain']:.4f} (tau={s['tau']})", flush=True)

    print("\n=== ensembles ===", flush=True)
    allw = np.ones(len(models), bool)
    s = score_set(allw)
    print(f"  ALL({len(models)})          organ={s['organ_acc']:.3f} dx={s['dx_acc']:.3f} "
          f"wf={s['workflow']:.4f} wf_abstain={s['workflow_abstain']:.4f} (tau={s['tau']})", flush=True)

    # also the best subset by greedy add (often better than all)
    order = list(range(len(models)))
    chosen, best = [], dict(workflow_abstain=-1)
    remaining = set(order)
    while remaining:
        cand_best, cand_i = None, None
        for i in remaining:
            w = np.zeros(len(models), bool); w[chosen + [i]] = True
            sc = score_set(w)
            if cand_best is None or sc["workflow_abstain"] > cand_best["workflow_abstain"]:
                cand_best, cand_i = sc, i
        if cand_best["workflow_abstain"] <= best["workflow_abstain"]:
            break
        chosen.append(cand_i); best = cand_best; remaining.discard(cand_i)
        names = [models[j][0] for j in chosen]
        print(f"  greedy+{models[cand_i][0]:16s} -> {names} "
              f"wf={best['workflow']:.4f} wf_abstain={best['workflow_abstain']:.4f}", flush=True)
    print(f"\nBEST ensemble: {[models[j][0] for j in chosen]} "
          f"wf_abstain={best['workflow_abstain']:.4f}  (single best f2_fuse_dxw=0.8143, oracle=0.889)", flush=True)


if __name__ == "__main__":
    main()
