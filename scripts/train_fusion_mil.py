#!/usr/bin/env python3
"""Train the MIL multi-task head on FUSED CONCH(512)+UNI2-h(1536) tile embeddings.

CONCH and UNI2-h are extracted from the SAME deterministic tiles per slide (sample_tiles
has no randomness), so tile i aligns across encoders and we can early-fuse by per-tile
concatenation -> 2048-d bags -> the same gated-attention MIL used by train_mil.py. Early
fusion lets the attention weigh tiles using the joint vision features of both foundation
models; UNI2-h (ViT-H pure-vision) is expected to sharpen fine-grained diagnosis where
frozen CONCH (ViT-B VL) plateaus (dx_acc ~0.69).

Reuses the deterministic split, dataset, collate, and the hierarchical/abstention eval
from train_mil.py so val workflow scores stay directly comparable to the CONCH-only runs
and the 0.889 oracle ceiling.

Outputs: artifacts/mil/<name>/{mil_head.pt,label_maps.json,metrics.json,history.json}
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/group/anantm-g00/REG2026")
sys.path.insert(0, "/group/anantm-g00/REG2026/scripts")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reg2026.labels import build_label_space, dx_label_to_organ_dx
from reg2026.templates import build_templates, apply_template
from reg2026 import metrics
from reg2026.mil import MILClassifier
from reg2026.gradings import build_grading_space, target_vector
# reuse the exact shared logic (split, dataset, padding collate, abstention eval)
from train_mil import hash_split, SlideDS, collate, evaluate
from torch.utils.data import Dataset

ROOT = "/group/anantm-g00/REG2026"
DATA = f"{ROOT}/Data/train_CoT.json"
EMB_CONCH = f"{ROOT}/artifacts/embeddings/conch/train"
EMB_UNI2H = f"{ROOT}/artifacts/embeddings/uni2h/train"
OUT_ROOT = f"{ROOT}/artifacts/mil"
IN_DIM = 2048  # 512 (CONCH) + 1536 (UNI2-h)

# Fusion configs. Richer 2048-d input -> a bit more capacity than the CONCH-only r1_reg.
CONFIGS = [
    dict(name="f0_fuse", lr=5e-4, hidden=768, attn_dim=384, dropout=0.40, wd=1e-4,
         organ_w=0.5, dx_w=1.0, smooth=0.1, epochs=80, bs=64, tile_drop=0.10, balanced=False),
    dict(name="f1_fuse_big", lr=4e-4, hidden=1024, attn_dim=512, dropout=0.45, wd=1.5e-4,
         organ_w=0.5, dx_w=1.0, smooth=0.1, epochs=90, bs=64, tile_drop=0.10, balanced=False),
    dict(name="f2_fuse_dxw", lr=5e-4, hidden=768, attn_dim=384, dropout=0.40, wd=1e-4,
         organ_w=0.4, dx_w=1.5, smooth=0.05, epochs=90, bs=64, tile_drop=0.10, balanced=False),
    # grading sub-heads: auxiliary multi-task supervision to sharpen dx (+ predict
    # template-substitutable grading fields). grade_w scales the summed masked grading CE.
    dict(name="f3_fuse_grade", lr=5e-4, hidden=768, attn_dim=384, dropout=0.40, wd=1e-4,
         organ_w=0.5, dx_w=1.0, smooth=0.1, epochs=90, bs=64, tile_drop=0.10, balanced=False,
         grading=True, grade_w=0.3),
]


class GradeSlideDS(Dataset):
    """Like train_mil.SlideDS but also returns a per-case grading target vector
    (len == #grading fields, -1 = field absent -> ignored in the loss)."""
    def __init__(self, cases, labels, cache, grade_targets, n_fields, tile_drop=0.0, train=False):
        self.inner = SlideDS(cases, labels, cache, tile_drop=tile_drop, train=train)
        self.grade_targets, self.n_fields = grade_targets, n_fields

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        feats, o, d = self.inner[i]
        cid = self.inner.items[i][0]
        g = self.grade_targets.get(cid, [-1] * self.n_fields)
        return feats, o, d, torch.tensor(g, dtype=torch.long)


def collate_grade(batch):
    feats = [b[0] for b in batch]
    organs = [b[1] for b in batch]; dxs = [b[2] for b in batch]
    g = torch.stack([b[3] for b in batch])               # (B, n_fields)
    x, mask, o, d = collate(list(zip(feats, organs, dxs)))
    return x, mask, o, d, g


def emb_paths(cid):
    base = cid[:-5] if cid.endswith(".tiff") else cid
    return os.path.join(EMB_CONCH, base + ".npy"), os.path.join(EMB_UNI2H, base + ".npy")


def load_fusion_cache(ids):
    """Per-tile concat of CONCH and UNI2-h. Includes a slide only if BOTH .npy exist and
    have matching tile counts (deterministic tiling guarantees alignment when both present)."""
    cache, n_skip_missing, n_skip_mismatch = {}, 0, 0
    for cid in ids:
        pc, pu = emb_paths(cid)
        if not (os.path.exists(pc) and os.path.exists(pu)):
            n_skip_missing += 1
            continue
        c = np.load(pc).astype(np.float32)
        u = np.load(pu).astype(np.float32)
        if c.shape[0] != u.shape[0]:
            n_skip_mismatch += 1
            continue
        cache[cid] = np.concatenate([c, u], axis=1)   # (N, 2048)
    return cache, n_skip_missing, n_skip_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="train on ALL data (deployment model)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    name = args.name or (cfg["name"] + ("_full" if args.full else ""))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    out_dir = os.path.join(OUT_ROOT, name); os.makedirs(out_dir, exist_ok=True)
    print(f"[{name}] FUSION config={cfg} full={args.full} device={dev}", flush=True)

    data = json.load(open(DATA))
    split_tr, split_va = hash_split(data)
    ls = build_label_space(data)
    organ_list, dx_list, labels = ls["organ_list"], ls["dx_list"], ls["labels"]
    n_organ, n_dx = len(organ_list), len(dx_list)
    organ_idx = {o: i for i, o in enumerate(organ_list)}
    dx_organ = [organ_idx[d.split("||")[0]] for d in dx_list]

    tr_cases = data if args.full else split_tr
    va_cases = split_va
    tpl = build_templates(tr_cases)

    t0 = time.time()
    cache, miss, mism = load_fusion_cache([c["id"] for c in data])
    print(f"fused cache: {len(cache)} slides (skipped {miss} missing one encoder, "
          f"{mism} tile-count mismatch) in {time.time()-t0:.0f}s", flush=True)
    if len(cache) < 1000:
        print("WARNING: <1000 fused slides — UNI2-h extraction is likely still running. "
              "Re-run when artifacts/embeddings/uni2h/train is complete.", flush=True)

    # optional grading sub-heads
    grading_dims = None
    grade_targets = {}
    if cfg.get("grading"):
        gs = build_grading_space(data)
        grading_dims = gs["dims"]
        grade_targets = {cid: target_vector(lab, grading_dims) for cid, lab in gs["labels"].items()}
        print(f"grading heads: {grading_dims} ({len(grade_targets)} graded cases)", flush=True)

    nw = min(16, max(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 4))
    if grading_dims:
        tr_ds = GradeSlideDS(tr_cases, labels, cache, grade_targets, len(grading_dims),
                             tile_drop=cfg["tile_drop"], train=True)
        tr_ld = DataLoader(tr_ds, batch_size=cfg["bs"], shuffle=True, collate_fn=collate_grade,
                           num_workers=nw, drop_last=True, persistent_workers=True)
    else:
        tr_ds = SlideDS(tr_cases, labels, cache, tile_drop=cfg["tile_drop"], train=True)
        tr_ld = DataLoader(tr_ds, batch_size=cfg["bs"], shuffle=True, collate_fn=collate,
                           num_workers=nw, drop_last=True, persistent_workers=True)
    va_ds = SlideDS(va_cases, labels, cache, train=False)   # val keeps 4-tuple for evaluate()
    print(f"train_ds={len(tr_ds)} val_ds={len(va_ds)}", flush=True)
    va_ld = DataLoader(va_ds, batch_size=cfg["bs"], shuffle=False, collate_fn=collate,
                       num_workers=nw, persistent_workers=True)

    dx_weight = None
    if cfg["balanced"]:
        cnt = np.ones(n_dx)
        for cid, lab in labels.items():
            cnt[lab["dx"]] += 1
        w = np.clip(cnt.sum() / (n_dx * cnt), 0.2, 5.0)
        dx_weight = torch.tensor(w, dtype=torch.float32, device=dev)

    mcfg = dict(in_dim=IN_DIM, hidden=cfg["hidden"], attn_dim=cfg["attn_dim"],
                dropout=cfg["dropout"], n_organ=n_organ, n_dx=n_dx, dx_organ=dx_organ,
                grading_dims=grading_dims)
    model = MILClassifier.from_config(mcfg).to(dev)
    ce_grade = nn.CrossEntropyLoss(ignore_index=-1)   # masks absent grading fields
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    ce_o = nn.CrossEntropyLoss(label_smoothing=cfg["smooth"])
    ce_d = nn.CrossEntropyLoss(label_smoothing=cfg["smooth"], weight=dx_weight)

    best = {"workflow": -1}
    history = []
    for ep in range(cfg["epochs"]):
        model.train(); tl = nb = 0
        for batch in tr_ld:
            if grading_dims:
                x, mask, o, d, g = batch
                g = g.to(dev)
            else:
                x, mask, o, d = batch
            x, mask, o, d = x.to(dev), mask.to(dev), o.to(dev), d.to(dev)
            out = model(x, mask)
            loss = cfg["organ_w"] * ce_o(out["organ"], o) + cfg["dx_w"] * ce_d(out["dx"], d)
            if grading_dims:
                # only add a field's loss when the batch has >=1 valid target for it;
                # an all-absent field would make CrossEntropyLoss average over 0 -> NaN.
                gl = 0.0
                for j, (f, _) in enumerate(grading_dims):
                    tgt = g[:, j]
                    if (tgt != -1).any():
                        gl = gl + ce_grade(out["grading"][f], tgt)
                loss = loss + cfg["grade_w"] * gl
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        oa, da, wf, wf_ab, tau = evaluate(model, va_ld, va_cases, labels, organ_list, dx_list, tpl, dev)
        history.append(dict(epoch=ep, loss=tl/max(nb,1), organ_acc=oa, dx_acc=da,
                            workflow=wf, workflow_abstain=wf_ab, tau=tau))
        print(f"[{name}] ep{ep:03d} loss={tl/max(nb,1):.3f} organ_acc={oa:.3f} "
              f"dx_acc={da:.3f} wf={wf:.4f} wf_abstain={wf_ab:.4f} (tau={tau})", flush=True)
        if wf_ab > best["workflow"]:
            best = dict(workflow=wf_ab, workflow_noabstain=wf, organ_acc=oa, dx_acc=da,
                        epoch=ep, abstain_tau=tau)
            torch.save({"config": mcfg, "state_dict": model.state_dict(), "abstain_tau": tau},
                       os.path.join(out_dir, "mil_head.pt"))
            json.dump({"organ": organ_list, "dx": dx_list, "dx_organ": dx_organ, "abstain_tau": tau},
                      open(os.path.join(out_dir, "label_maps.json"), "w"))
    json.dump(best, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    print(f"[{name}] DONE best={best} (CONCH-only r1_reg=0.794; oracle=0.889)", flush=True)


if __name__ == "__main__":
    main()
