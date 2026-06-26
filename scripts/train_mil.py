#!/usr/bin/env python3
"""Train the MIL multi-task head (organ + dx) on precomputed CONCH tile embeddings.

Targets come from reg2026.labels (organ 10-way, dx <organ>||<diagnosis> bucketed). The
train/val split is the SAME deterministic hash split used by scripts/eval_oracle.py
(seed=0, frac=0.8) so the validation workflow score is directly comparable to the oracle
ceilings (oracle organ+dx = 0.89). Each epoch we decode predicted (organ, dx) -> template
-> workflow_score, and keep the checkpoint with the best val workflow total.

Sweep: --config 0..3 picks a hyperparameter set; run as a SLURM array to use 4 GPUs.
Outputs: artifacts/mil/<name>/{mil_head.pt,label_maps.json,metrics.json,history.json}
"""
import argparse, json, os, sys, time, glob
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from reg2026.labels import build_label_space, dx_label_to_organ_dx
from reg2026.templates import build_templates, apply_template
from reg2026 import metrics

ROOT = "/group/anantm-g00/REG2026"
DATA = f"{ROOT}/Data/train_CoT.json"
EMB_DIR = f"{ROOT}/artifacts/embeddings/conch/train"
OUT_ROOT = f"{ROOT}/artifacts/mil"
IN_DIM = 512

CONFIGS = [
    dict(name="r0_base", lr=1e-3, hidden=512, attn_dim=256, dropout=0.25, wd=1e-4,
         organ_w=0.5, dx_w=1.0, smooth=0.1, epochs=60, bs=64, tile_drop=0.0, balanced=False),
    dict(name="r1_reg",  lr=5e-4, hidden=512, attn_dim=384, dropout=0.40, wd=1e-4,
         organ_w=0.5, dx_w=1.0, smooth=0.1, epochs=80, bs=64, tile_drop=0.10, balanced=False),
    dict(name="r2_big",  lr=5e-4, hidden=768, attn_dim=512, dropout=0.30, wd=1e-4,
         organ_w=0.5, dx_w=1.0, smooth=0.1, epochs=80, bs=64, tile_drop=0.10, balanced=False),
    dict(name="r3_bal",  lr=1e-3, hidden=512, attn_dim=256, dropout=0.25, wd=5e-5,
         organ_w=0.5, dx_w=1.5, smooth=0.05, epochs=70, bs=128, tile_drop=0.0, balanced=True),
]


def hash_split(cases, frac=0.8, seed=0):
    # Deterministic across processes: Python's builtin hash() is salted per process
    # (PYTHONHASHSEED), which would give a different split every run. Use a stable digest.
    import hashlib
    tr, va = [], []
    for c in cases:
        digest = hashlib.md5(f"{seed}:{c['id']}".encode()).hexdigest()
        h = (int(digest[:8], 16) & 0xFFFFFFFF) / 0xFFFFFFFF
        (tr if h < frac else va).append(c)
    return tr, va


def emb_path(cid):
    # label ids carry a .tiff suffix; embedding files do not
    base = cid[:-5] if cid.endswith(".tiff") else cid
    return os.path.join(EMB_DIR, base + ".npy")


class SlideDS(Dataset):
    def __init__(self, cases, labels, cache, tile_drop=0.0, train=False):
        self.items, self.cache = [], cache
        self.tile_drop, self.train = tile_drop, train
        for c in cases:
            cid = c["id"]
            if cid in labels and cid in cache:
                lab = labels[cid]
                self.items.append((cid, lab["organ"], lab["dx"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, o, d = self.items[i]
        feats = self.cache[cid]                       # (N, D) float32
        if self.train and self.tile_drop > 0 and feats.shape[0] > 8:
            keep = np.random.rand(feats.shape[0]) >= self.tile_drop
            if keep.sum() >= 4:
                feats = feats[keep]
        return torch.from_numpy(feats), o, d


def collate(batch):
    feats, organs, dxs = zip(*batch)
    N = max(f.shape[0] for f in feats)
    B, D = len(feats), feats[0].shape[1]
    x = torch.zeros(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for i, f in enumerate(feats):
        n = f.shape[0]
        x[i, :n] = f
        mask[i, :n] = True
    return x, mask, torch.tensor(organs), torch.tensor(dxs)


def load_cache(ids):
    cache = {}
    for cid in ids:
        p = emb_path(cid)
        if os.path.exists(p):
            cache[cid] = np.load(p).astype(np.float32)
    return cache


TAU_GRID = [round(t, 3) for t in np.linspace(0.0, 0.95, 20)]


@torch.no_grad()
def evaluate(model, loader, va_cases, labels, organ_list, dx_list, tpl, dev):
    """Hierarchical (organ-conditioned) eval with confidence-based abstention tuning.
    Returns organ_acc, dx_acc, workflow (no abstain), workflow_abstain (best tau), best_tau."""
    model.eval()
    id_items = loader.dataset.items
    cases_by_id = {c["id"]: c for c in va_cases}
    no = nd = tot = 0
    idx = 0
    per_case = []   # (conf, s_dx, s_organ_only)
    for x, mask, o, d in loader:
        x, mask = x.to(dev), mask.to(dev)
        out = model(x, mask)
        po_t = out["organ"].argmax(-1)
        dx_masked = model.masked_dx_logits(out["dx"], po_t)   # restrict to predicted organ
        pd_t = dx_masked.argmax(-1)
        conf_t = torch.softmax(dx_masked, -1).max(-1).values
        po = po_t.cpu().tolist(); pd = pd_t.cpu().tolist(); conf = conf_t.cpu().tolist()
        no += (po_t.cpu() == o).sum().item(); nd += (pd_t.cpu() == d).sum().item(); tot += len(o)
        for k in range(len(po)):
            cid = id_items[idx][0]; idx += 1
            gt = cases_by_id[cid]["chain-of-thought"]
            organ_str = organ_list[po[k]]
            _, dx_str = dx_label_to_organ_dx(dx_list[pd[k]])
            s_dx = metrics.workflow_score(apply_template({"organ": organ_str, "dx": dx_str}, tpl), gt)["total"]
            s_org = metrics.workflow_score(apply_template({"organ": organ_str, "dx": None}, tpl), gt)["total"]
            per_case.append((conf[k], s_dx, s_org))
    organ_acc, dx_acc = no / tot, nd / tot
    wf_noabstain = float(np.mean([p[1] for p in per_case]))
    # sweep abstention threshold: use dx prediction if conf>=tau else back off to organ template
    best_wf, best_tau = wf_noabstain, 0.0
    for tau in TAU_GRID:
        m = float(np.mean([(p[1] if p[0] >= tau else p[2]) for p in per_case]))
        if m > best_wf:
            best_wf, best_tau = m, tau
    return organ_acc, dx_acc, wf_noabstain, best_wf, best_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="train on ALL data (deployment model); monitor on the held-out subset")
    ap.add_argument("--name", default=None, help="override output dir name")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    name = args.name or (cfg["name"] + ("_full" if args.full else "_hier"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    out_dir = os.path.join(OUT_ROOT, name); os.makedirs(out_dir, exist_ok=True)
    print(f"[{name}] config={cfg} full={args.full} device={dev}", flush=True)

    data = json.load(open(DATA))
    split_tr, split_va = hash_split(data)
    ls = build_label_space(data)
    organ_list, dx_list, labels = ls["organ_list"], ls["dx_list"], ls["labels"]
    n_organ, n_dx = len(organ_list), len(dx_list)
    # hierarchical map: organ index that each dx class belongs to (dx_list entries are "<organ>||...")
    organ_idx = {o: i for i, o in enumerate(organ_list)}
    dx_organ = [organ_idx[d.split("||")[0]] for d in dx_list]
    # --full: train on everything for deployment, still monitor on the held-out subset.
    tr_cases = data if args.full else split_tr
    va_cases = split_va
    tpl = build_templates(tr_cases)
    print(f"train={len(tr_cases)} val={len(va_cases)} n_organ={n_organ} n_dx={n_dx}", flush=True)

    # cache embeddings into RAM (skip ids without a .npy yet)
    t0 = time.time()
    all_ids = [c["id"] for c in data]
    cache = load_cache(all_ids)
    print(f"cached {len(cache)}/{len(all_ids)} embeddings in {time.time()-t0:.0f}s", flush=True)

    tr_ds = SlideDS(tr_cases, labels, cache, tile_drop=cfg["tile_drop"], train=True)
    va_ds = SlideDS(va_cases, labels, cache, train=False)
    print(f"train_ds={len(tr_ds)} val_ds={len(va_ds)}", flush=True)
    nw = min(16, max(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 4))
    tr_ld = DataLoader(tr_ds, batch_size=cfg["bs"], shuffle=True, collate_fn=collate,
                       num_workers=nw, drop_last=True, persistent_workers=True)
    va_ld = DataLoader(va_ds, batch_size=cfg["bs"], shuffle=False, collate_fn=collate,
                       num_workers=nw, persistent_workers=True)

    # dx class weights (balanced run)
    dx_weight = None
    if cfg["balanced"]:
        cnt = np.ones(n_dx)
        for cid, lab in labels.items():
            cnt[lab["dx"]] += 1
        w = (cnt.sum() / (n_dx * cnt)); w = np.clip(w, 0.2, 5.0)
        dx_weight = torch.tensor(w, dtype=torch.float32, device=dev)

    from reg2026.mil import MILClassifier
    mcfg = dict(in_dim=IN_DIM, hidden=cfg["hidden"], attn_dim=cfg["attn_dim"],
                dropout=cfg["dropout"], n_organ=n_organ, n_dx=n_dx, dx_organ=dx_organ)
    model = MILClassifier.from_config(mcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    ce_o = nn.CrossEntropyLoss(label_smoothing=cfg["smooth"])
    ce_d = nn.CrossEntropyLoss(label_smoothing=cfg["smooth"], weight=dx_weight)

    best = {"workflow": -1}
    history = []
    for ep in range(cfg["epochs"]):
        model.train(); tl = 0; nb = 0
        for x, mask, o, d in tr_ld:
            x, mask, o, d = x.to(dev), mask.to(dev), o.to(dev), d.to(dev)
            out = model(x, mask)
            # Train the dx head flat (label smoothing is incompatible with -inf masking);
            # the hierarchical organ-conditioned mask is applied only at eval/inference.
            loss = cfg["organ_w"] * ce_o(out["organ"], o) + cfg["dx_w"] * ce_d(out["dx"], d)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        oa, da, wf, wf_ab, tau = evaluate(model, va_ld, va_cases, labels, organ_list, dx_list, tpl, dev)
        history.append(dict(epoch=ep, loss=tl/max(nb,1), organ_acc=oa, dx_acc=da,
                            workflow=wf, workflow_abstain=wf_ab, tau=tau))
        print(f"[{name}] ep{ep:03d} loss={tl/max(nb,1):.3f} organ_acc={oa:.3f} "
              f"dx_acc={da:.3f} wf={wf:.4f} wf_abstain={wf_ab:.4f} (tau={tau})", flush=True)
        # select on abstention-tuned workflow (the deployable metric)
        if wf_ab > best["workflow"]:
            best = dict(workflow=wf_ab, workflow_noabstain=wf, organ_acc=oa, dx_acc=da,
                        epoch=ep, abstain_tau=tau)
            torch.save({"config": mcfg, "state_dict": model.state_dict(), "abstain_tau": tau},
                       os.path.join(out_dir, "mil_head.pt"))
            json.dump({"organ": organ_list, "dx": dx_list, "dx_organ": dx_organ,
                       "abstain_tau": tau},
                      open(os.path.join(out_dir, "label_maps.json"), "w"))
    json.dump(best, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    print(f"[{name}] DONE best={best} (oracle ceiling=0.889)", flush=True)


if __name__ == "__main__":
    main()
