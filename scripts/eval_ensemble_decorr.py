#!/usr/bin/env python3
"""DECORRELATED ensemble test — does ensembling ENCODER-DIVERSE models help dx?

The prior eval_ensemble.py could only average heads that share the SAME 2048-d fused
features (f0/f1/f2/f3 + seeds) -> highly correlated -> +0.0004 = noise. This script
ensembles models with GENUINELY DIFFERENT feature spaces:
  - CONCH-only        (512)   r1_reg
  - CONCH+UNI2-h      (2048)  f2_fuse_dxw, f1_fuse_big
  - +Virchow2         (4608)  f2_fuse_dxw_v2
and judges on RAW dx-accuracy (exact, no proxy offset) with a BOOTSTRAP CI on the
ensemble-vs-best-single gain, plus decorrelation diagnostics (pairwise agreement and
"at-least-one-correct" oracle ceiling). It also reports the DEPLOYABLE pair
{r1_reg + f2_fuse_dxw} which needs NO extra in-container encoder.
"""
import sys, json, os
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np, torch

from reg2026.labels import build_label_space, dx_label_to_organ_dx
from reg2026.mil import MILClassifier
from scripts.train_mil import hash_split

ROOT = "/group/anantm-g00/REG2026"
DATA = f"{ROOT}/Data/train_CoT.json"
EMB = {"conch": f"{ROOT}/artifacts/embeddings/conch/train",
       "uni2h": f"{ROOT}/artifacts/embeddings/uni2h/train",
       "virchow2": f"{ROOT}/artifacts/embeddings/virchow2/train"}

# (run name, list of encoder views to concat in order)
MODELS = [
    ("r1_reg",          ["conch"]),
    ("f2_fuse_dxw",     ["conch", "uni2h"]),
    ("f1_fuse_big",     ["conch", "uni2h"]),
    ("f2_fuse_dxw_v2",  ["conch", "uni2h", "virchow2"]),
]


def load_views(cid):
    """Return dict enc->(N,d) with aligned N, or None if any missing/mismatch."""
    base = cid[:-5] if cid.endswith(".tiff") else cid
    out = {}
    for e, d in EMB.items():
        p = os.path.join(d, base + ".npy")
        if not os.path.exists(p):
            return None
        a = np.load(p).astype(np.float32)
        if a.shape[0] <= 1:           # placeholder
            return None
        out[e] = a
    if len({a.shape[0] for a in out.values()}) != 1:
        return None
    return out


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


@torch.no_grad()
def model_probs(model, feats):
    x = torch.from_numpy(feats).unsqueeze(0)
    mask = torch.ones(1, feats.shape[0], dtype=torch.bool)
    out = model(x, mask)
    return softmax(out["organ"].numpy())[0], softmax(out["dx"].numpy())[0]


def main():
    data = json.load(open(DATA))
    tr, va = hash_split(data)
    ls = build_label_space(data)
    organ_list, dx_list, labels = ls["organ_list"], ls["dx_list"], ls["labels"]
    organ_idx = {o: i for i, o in enumerate(organ_list)}
    dx_organ = np.array([organ_idx[d.split("||")[0]] for d in dx_list])

    models = []
    for name, views in MODELS:
        ck = torch.load(f"{ROOT}/artifacts/mil/{name}/mil_head.pt", map_location="cpu", weights_only=False)
        m = MILClassifier.from_config(ck["config"]); m.load_state_dict(ck["state_dict"]); m.eval()
        models.append((name, views, m))
    print(f"models: {[n for n,_,_ in models]}", flush=True)

    # gather per-slide per-model dx softmax (organ from the fusion model only — organ is ~0.96 solid)
    O = np.full((0,), 0)
    DP, OP, DGT, OGT = [], [], [], []     # lists per slide: dprobs[M,n_dx], oprobs[M,n_organ]
    n_skip = 0
    for c in va:
        cid = c["id"]
        if cid not in labels:
            continue
        v = load_views(cid)
        if v is None:
            n_skip += 1; continue
        dps, ops = [], []
        for name, views, m in models:
            feats = np.concatenate([v[e] for e in views], axis=1)
            op, dp = model_probs(m, feats)
            ops.append(op); dps.append(dp)
        DP.append(np.array(dps)); OP.append(np.array(ops))
        DGT.append(labels[cid]["dx"]); OGT.append(labels[cid]["organ"])
    DP = np.array(DP); OP = np.array(OP)            # (S, M, n_dx), (S, M, n_organ)
    DGT = np.array(DGT); OGT = np.array(OGT)
    S, M = DP.shape[0], DP.shape[1]
    print(f"held-out slides used: {S} (skipped {n_skip} missing/placeholder)\n", flush=True)

    def decode(dp_avg, op_avg):
        """dp_avg (S,n_dx), op_avg (S,n_organ) -> (organ_pred, dx_pred) with hier mask."""
        po = op_avg.argmax(1)
        masked = np.where(dx_organ[None, :] == po[:, None], dp_avg, -1.0)
        pd = masked.argmax(1)
        return po, pd

    def dxacc(pd):
        return float((pd == DGT).mean())

    def bal_dxacc(pd):
        accs = []
        for cl in np.unique(DGT):
            sel = DGT == cl
            if sel.sum() >= 3:
                accs.append((pd[sel] == cl).mean())
        return float(np.mean(accs))

    # per-model predictions (each uses its OWN organ prob)
    print("=== single models (raw dx-acc, exact) ===")
    single_pd = []
    for i, (name, _, _) in enumerate(models):
        po, pd = decode(DP[:, i], OP[:, i])
        single_pd.append(pd)
        print(f"  {name:16s} organ_acc={float((po==OGT).mean()):.4f} dx_acc={dxacc(pd):.4f} bal_dx={bal_dxacc(pd):.4f}")
    single_pd = np.array(single_pd)     # (M, S)
    best_single_i = int(np.argmax([dxacc(single_pd[i]) for i in range(M)]))
    best_single_pd = single_pd[best_single_i]
    print(f"  best single = {models[best_single_i][0]} (dx_acc={dxacc(best_single_pd):.4f})\n")

    # decorrelation diagnostics
    print("=== decorrelation diagnostics ===")
    print("  pairwise dx-prediction agreement:")
    for i in range(M):
        for j in range(i+1, M):
            agree = float((single_pd[i] == single_pd[j]).mean())
            print(f"    {models[i][0]:16s} vs {models[j][0]:16s}: {agree:.3f}")
    atleast1 = np.zeros(S, bool)
    for i in range(M):
        atleast1 |= (single_pd[i] == DGT)
    print(f"  AT-LEAST-ONE-correct (ensemble ORACLE ceiling) = {float(atleast1.mean()):.4f}")
    print(f"  best single dx_acc                              = {dxacc(best_single_pd):.4f}")
    print(f"  -> exploitable decorrelation headroom           = {float(atleast1.mean())-dxacc(best_single_pd):+.4f}\n")

    # ensembles: average softmax. organ from mean of organ probs across fusion models.
    def ens(idxs):
        dp = DP[:, idxs].mean(1)
        op = OP[:, idxs].mean(1)
        return decode(dp, op)

    sets = {
        "ALL-4": list(range(M)),
        "deployable {r1_reg+f2}": [0, 1],
        "encoder-diverse {r1_reg+f2+v2}": [0, 1, 3],
        "fusion-only {f2+f1+v2}": [1, 2, 3],
    }
    print("=== ensembles (mean softmax) ===")
    results = {}
    for tag, idxs in sets.items():
        po, pd = ens(idxs)
        results[tag] = pd
        print(f"  {tag:34s} dx_acc={dxacc(pd):.4f} bal_dx={bal_dxacc(pd):.4f} "
              f"(Δ vs best single {dxacc(pd)-dxacc(best_single_pd):+.4f})")

    # bootstrap CI on (best ensemble - best single) dx-acc gain
    best_ens_tag = max(results, key=lambda t: dxacc(results[t]))
    best_ens_pd = results[best_ens_tag]
    print(f"\n=== bootstrap CI: ({best_ens_tag}) - (best single {models[best_single_i][0]}) ===")
    rng = np.random.RandomState(0)
    diffs = []
    corr_e = (best_ens_pd == DGT).astype(float)
    corr_s = (best_single_pd == DGT).astype(float)
    for _ in range(2000):
        idx = rng.randint(0, S, S)
        diffs.append(corr_e[idx].mean() - corr_s[idx].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  mean Δdx_acc = {diffs.mean():+.4f}   95% CI = [{lo:+.4f}, {hi:+.4f}]   "
          f"P(Δ>0) = {float((diffs>0).mean()):.3f}")
    print(f"  -> {'SIGNAL (CI excludes 0)' if lo>0 else 'NOT distinguishable from noise (CI includes 0)'}")


if __name__ == "__main__":
    main()
