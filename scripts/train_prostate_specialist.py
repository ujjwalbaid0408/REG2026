#!/usr/bin/env python3
"""Prostate tumor-detection SPECIALIST head.

The main fusion model's single biggest concentrated error bucket is the prostate
binary decision 'No tumor present' <-> 'Acinar adenocarcinoma' (43/2253 held-out
slides; whatif ROI = +0.011 workflow if solved). Prostate is essentially binary
(1595 Acinar adenocarcinoma vs 820 No tumor), and tumor detection is exactly what
attention-MIL does best -- so a dedicated binary head over the SAME frozen fusion
features (2048-d CONCH+UNI2-h) should beat the 134-way head's ~0.91 prostate split.

Trains on prostate slides only, SAME hash_split as the main model (no leakage).
At inference (separate integration) its decision OVERRIDES the main model's choice
ONLY when the main model predicts Prostate + {No tumor, Acinar adenocarcinoma}.
Saves artifacts/mil/prostate_specialist/spec_head.pt.
"""
import json, os, sys, argparse
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import scripts.train_fusion_mil as F
from scripts.train_mil import hash_split
from reg2026.labels import build_label_space, dx_label_to_organ_dx, field, ORGAN_Q
from reg2026.canon import canon
from reg2026.mil import GatedAttentionMIL

ROOT = "/group/anantm-g00/REG2026"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NO_TUMOR = "No tumor present"
ACINAR = "Acinar adenocarcinoma"


def dx_name(cot):
    for st in cot:
        if "diagnosis" in canon(st.get("question")):
            return st.get("answer")
    return None


class ProstateDS(Dataset):
    def __init__(self, cases, cache, train=False, tile_drop=0.0):
        self.items, self.cache, self.train, self.tile_drop = [], cache, train, tile_drop
        for c in cases:
            cid = c["id"]
            if cid not in cache:
                continue
            if (field(c["chain-of-thought"], ORGAN_Q) or "") != "Prostate":
                continue
            nm = dx_name(c["chain-of-thought"])
            if nm is None:
                continue
            y = 0 if nm == NO_TUMOR else 1          # tumor present = 1
            self.items.append((cid, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, y = self.items[i]
        feats = self.cache[cid]
        if self.train and self.tile_drop > 0 and feats.shape[0] > 8:
            keep = np.random.rand(feats.shape[0]) >= self.tile_drop
            if keep.sum() >= 4:
                feats = feats[keep]
        return torch.from_numpy(feats), y


def collate(batch):
    feats, ys = zip(*batch)
    N = max(f.shape[0] for f in feats)
    D = feats[0].shape[1]
    x = torch.zeros(len(feats), N, D)
    m = torch.zeros(len(feats), N, dtype=torch.bool)
    for i, f in enumerate(feats):
        x[i, :f.shape[0]] = f
        m[i, :f.shape[0]] = True
    return x, m, torch.tensor(ys, dtype=torch.long)


class ProstateSpecialist(nn.Module):
    """Gated-attention pool + top-k instance-max branch (tumor = any focal tumor tile)."""
    def __init__(self, in_dim=2048, hidden=256, attn_dim=128, dropout=0.25, topk=8):
        super().__init__()
        self.mil = GatedAttentionMIL(in_dim, hidden, attn_dim, dropout)
        self.inst = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Dropout(dropout))
        self.head = nn.Linear(hidden + 1, 2)
        self.topk = topk

    def forward(self, x, mask=None):
        pooled, _ = self.mil(x, mask)                 # (B, hidden)
        z = self.trunk(pooled)
        s = self.inst(x).squeeze(-1)                  # (B, N) per-tile tumor score
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        k = min(self.topk, s.shape[1])
        topk = torch.topk(s, k, dim=1).values         # (B, k)
        topk = topk.masked_fill(torch.isinf(topk), 0.0).mean(dim=1, keepdim=True)  # (B,1)
        return self.head(torch.cat([z, topk], dim=1))


def evaluate(model, ld):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, m, y in ld:
            logit = model(x.to(DEVICE), m.to(DEVICE)).cpu()
            yp.append(logit.argmax(-1)); yt.append(y)
    yt = torch.cat(yt).numpy(); yp = torch.cat(yp).numpy()
    acc = (yt == yp).mean()
    # balanced acc
    baccs = []
    for c in (0, 1):
        sel = yt == c
        if sel.any():
            baccs.append((yp[sel] == c).mean())
    return acc, float(np.mean(baccs)), yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--name", default="prostate_specialist")
    args = ap.parse_args()

    data = json.load(open(F.DATA))
    tr, va = hash_split(data)
    ids = [c["id"] for c in data
           if (field(c["chain-of-thought"], ORGAN_Q) or "") == "Prostate"]
    cache, miss, mism = F.load_fusion_cache(ids)
    print(f"# prostate fused cache: {len(cache)} (miss {miss} / mismatch {mism}), device={DEVICE}")

    tr_ds = ProstateDS(tr, cache, train=True, tile_drop=0.1)
    va_ds = ProstateDS(va, cache, train=False)
    ntr1 = sum(y for _, y in tr_ds.items);
    print(f"# train {len(tr_ds)} (tumor {ntr1}/{len(tr_ds)-ntr1} no-tumor)  val {len(va_ds)}")
    tr_ld = DataLoader(tr_ds, batch_size=24, shuffle=True, collate_fn=collate, num_workers=4)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=4)

    model = ProstateSpecialist(in_dim=cache[ids[0]].shape[1]).to(DEVICE)
    # class weight: tumor is ~2x no-tumor -> upweight no-tumor
    n1 = ntr1; n0 = len(tr_ds) - ntr1
    w = torch.tensor([len(tr_ds)/(2*n0), len(tr_ds)/(2*n1)], dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = {"bacc": -1}
    for ep in range(args.epochs):
        model.train()
        for x, m, y in tr_ld:
            opt.zero_grad()
            loss = crit(model(x.to(DEVICE), m.to(DEVICE)), y.to(DEVICE))
            loss.backward(); opt.step()
        sched.step()
        acc, bacc, yt, yp = evaluate(model, va_ld)
        tag = ""
        if bacc > best["bacc"]:
            best = {"acc": float(acc), "bacc": float(bacc), "ep": ep,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
            tag = " *"
        print(f"ep{ep:02d} loss{loss.item():.3f} val_acc{acc:.4f} bal_acc{bacc:.4f}{tag}", flush=True)

    out_dir = os.path.join(F.OUT_ROOT, args.name); os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": best["state"], "in_dim": int(cache[ids[0]].shape[1]),
                "best": {k: best[k] for k in ("acc", "bacc", "ep")}},
               os.path.join(out_dir, "spec_head.pt"))
    print(f"\n# BEST prostate specialist: acc={best['acc']:.4f} bal_acc={best['bacc']:.4f} "
          f"(ep{best['ep']}). Baseline main-model prostate binary ~0.912.")


if __name__ == "__main__":
    main()
