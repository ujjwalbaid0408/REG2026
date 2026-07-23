#!/usr/bin/env python3
"""LoRA fine-tune CONCH end-to-end with the fusion MIL head -- the ceiling-breaker.

Frozen-feature fusion plateaus at dx_acc~0.737 (workflow 0.814). Here we unfreeze CONCH via
LoRA adapters (0.44M params, 0.11% of the backbone) and train end-to-end through the encoder
so the tile representation itself adapts to the diagnosis task. UNI2-h stays frozen (cached
embeddings); only CONCH is adapted, then concatenated to UNI2-h -> 2048-d -> gated-attn MIL.

Per slide we read cached raw tiles (artifacts/tiles/train/<id>.npy, deterministic, aligned to
the cached embeddings) and the cached UNI2-h embedding; tile i lines up across both. Training
subsamples K tiles/slide for compute; eval uses all tiles. Same hash split, template engine,
hierarchical dx masking and abstention sweep as train_fusion_mil so the workflow score is
directly comparable (frozen fusion f2_fuse_dxw=0.814, oracle=0.889).

Outputs: artifacts/mil/<name>/{lora.pt, mil_head.pt, label_maps.json, metrics.json, history.json}
"""
import argparse, json, os, sys, time, random
sys.path.insert(0, "/group/anantm-g00/REG2026")
sys.path.insert(0, "/group/anantm-g00/REG2026/scripts")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from reg2026.labels import build_label_space, dx_label_to_organ_dx
from reg2026.templates import build_templates, apply_template
from reg2026 import metrics
from reg2026.mil import MILClassifier
from reg2026.encoder import load_encoder
from reg2026.lora import inject_lora, lora_state_dict
from train_mil import hash_split

ROOT = "/group/anantm-g00/REG2026"
DATA = f"{ROOT}/Data/train_CoT.json"
TILES = f"{ROOT}/artifacts/tiles/train"
EMB_UNI2H = f"{ROOT}/artifacts/embeddings/uni2h/train"
OUT_ROOT = f"{ROOT}/artifacts/mil"
TAU_GRID = [round(t, 3) for t in np.linspace(0.0, 0.95, 20)]


def base_id(cid):
    return cid[:-5] if cid.endswith(".tiff") else cid


class TileBagDS(Dataset):
    """Returns (tiles uint8 [n,224,224,3], uni2h [n,1536], organ, dx). Includes a slide only
    if both the tile cache and the UNI2-h embedding exist with matching tile counts."""
    def __init__(self, cases, labels, k_tiles=64, train=False):
        self.items, self.k, self.train = [], k_tiles, train
        for c in cases:
            cid = c["id"]
            if cid not in labels:
                continue
            b = base_id(cid)
            tp, up = f"{TILES}/{b}.npy", f"{EMB_UNI2H}/{b}.npy"
            if os.path.exists(tp) and os.path.exists(up):
                self.items.append((cid, b, labels[cid]["organ"], labels[cid]["dx"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, b, o, d = self.items[i]
        tiles = np.load(f"{TILES}/{b}.npy")              # (N,224,224,3) uint8
        uni = np.load(f"{EMB_UNI2H}/{b}.npy").astype(np.float32)  # (N,1536)
        n = min(tiles.shape[0], uni.shape[0])
        tiles, uni = tiles[:n], uni[:n]
        if self.train and n > self.k:
            idx = np.sort(np.random.choice(n, self.k, replace=False))
            tiles, uni = tiles[idx], uni[idx]
        return torch.from_numpy(tiles), torch.from_numpy(uni), o, d


def collate(batch):
    tiles, unis, o, d = zip(*batch)
    N = max(t.shape[0] for t in tiles)
    B = len(tiles)
    xt = torch.zeros(B, N, 224, 224, 3, dtype=torch.uint8)
    xu = torch.zeros(B, N, unis[0].shape[1], dtype=torch.float32)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for i, (t, u) in enumerate(zip(tiles, unis)):
        n = t.shape[0]
        xt[i, :n] = t; xu[i, :n] = u; mask[i, :n] = True
    return xt, xu, mask, torch.tensor(o), torch.tensor(d)


class LoRAFusionMIL(nn.Module):
    """CONCH(+LoRA) on tiles -> 512, concat frozen UNI2-h cache -> 2048 -> gated-attn MIL."""
    def __init__(self, conch, head, enc_chunk=256):
        super().__init__()
        self.conch = conch
        self.head = head
        self.enc_chunk = enc_chunk
        mean, std = conch._emb_mean, conch._emb_std
        self.register_buffer("emb_mean", mean.clone())
        self.register_buffer("emb_std", std.clone())

    def encode_tiles(self, xt, mask):
        # xt: (B,N,224,224,3) uint8 -> (B,N,512)
        B, N = xt.shape[0], xt.shape[1]
        x = xt.permute(0, 1, 4, 2, 3).reshape(B * N, 3, 224, 224).float().div_(255.0)
        x = (x - self.emb_mean) / self.emb_std
        feats = []
        for i in range(0, x.shape[0], self.enc_chunk):
            feats.append(self.conch(x[i:i + self.enc_chunk]))
        return torch.cat(feats).view(B, N, -1)

    def forward(self, xt, xu, mask):
        c = self.encode_tiles(xt, mask)                  # (B,N,512)
        fused = torch.cat([c, xu], dim=-1)               # (B,N,2048)
        return self.head(fused, mask)

    def masked_dx_logits(self, dx_logits, organ_idx):
        return self.head.masked_dx_logits(dx_logits, organ_idx)


@torch.no_grad()
def evaluate(model, loader, va_cases, labels, organ_list, dx_list, tpl, dev):
    model.eval()
    id_items = loader.dataset.items
    cases_by_id = {c["id"]: c for c in va_cases}
    no = nd = tot = 0; idx = 0; per_case = []
    for xt, xu, mask, o, d in loader:
        xt, xu, mask = xt.to(dev), xu.to(dev), mask.to(dev)
        with torch.cuda.amp.autocast(enabled=dev.startswith("cuda")):
            out = model(xt, xu, mask)
        po_t = out["organ"].float().argmax(-1)
        dx_masked = model.masked_dx_logits(out["dx"].float(), po_t)
        pd_t = dx_masked.argmax(-1)
        conf_t = torch.softmax(dx_masked, -1).max(-1).values
        po, pd, conf = po_t.cpu().tolist(), pd_t.cpu().tolist(), conf_t.cpu().tolist()
        no += (po_t.cpu() == o).sum().item(); nd += (pd_t.cpu() == d).sum().item(); tot += len(o)
        for k in range(len(po)):
            cid = id_items[idx][0]; idx += 1
            gt = cases_by_id[cid]["chain-of-thought"]
            organ_str = organ_list[po[k]]
            _, dx_str = dx_label_to_organ_dx(dx_list[pd[k]])
            s_dx = metrics.workflow_score(apply_template({"organ": organ_str, "dx": dx_str}, tpl), gt)["total"]
            s_org = metrics.workflow_score(apply_template({"organ": organ_str, "dx": None}, tpl), gt)["total"]
            per_case.append((conf[k], s_dx, s_org))
    oa, da = no / tot, nd / tot
    wf = float(np.mean([p[1] for p in per_case]))
    best_wf, best_tau = wf, 0.0
    for tau in TAU_GRID:
        m = float(np.mean([(p[1] if p[0] >= tau else p[2]) for p in per_case]))
        if m > best_wf:
            best_wf, best_tau = m, tau
    return oa, da, wf, best_wf, best_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lora_conch_fuse")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--k-tiles", type=int, default=64)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--attn-dim", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.40)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--dx-w", type=float, default=1.5)
    ap.add_argument("--organ-w", type=float, default=0.4)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--warm-head", default=None, help="init MIL head from this run's mil_head.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    out_dir = os.path.join(OUT_ROOT, args.name); os.makedirs(out_dir, exist_ok=True)
    print(f"[{args.name}] LoRA fine-tune CONCH+fuse  dev={dev}  args={vars(args)}", flush=True)

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

    tr_ds = TileBagDS(tr_cases, labels, k_tiles=args.k_tiles, train=True)
    va_ds = TileBagDS(va_cases, labels, train=False)
    print(f"train_ds={len(tr_ds)} val_ds={len(va_ds)} (need tile cache + uni2h cache)", flush=True)
    nw = min(12, max(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 2))
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, collate_fn=collate,
                       num_workers=nw, drop_last=True, persistent_workers=True, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=max(2, args.bs), shuffle=False, collate_fn=collate,
                       num_workers=nw, persistent_workers=True, pin_memory=True)

    conch, _ = load_encoder("conch", device=dev)
    n_adapt = inject_lora(conch, r=args.r, alpha=args.alpha, dropout=0.05)
    print(f"LoRA adapted {n_adapt} layers; trainable encoder params "
          f"{sum(p.numel() for p in conch.parameters() if p.requires_grad)/1e6:.3f}M", flush=True)
    mcfg = dict(in_dim=2048, hidden=args.hidden, attn_dim=args.attn_dim, dropout=args.dropout,
                n_organ=n_organ, n_dx=n_dx, dx_organ=dx_organ)
    head = MILClassifier.from_config(mcfg)
    if args.warm_head:
        ck = torch.load(f"{OUT_ROOT}/{args.warm_head}/mil_head.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(ck["state_dict"]); print(f"warm-started head from {args.warm_head}", flush=True)
    model = LoRAFusionMIL(conch, head).to(dev)

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce_o = nn.CrossEntropyLoss(label_smoothing=args.smooth)
    ce_d = nn.CrossEntropyLoss(label_smoothing=args.smooth)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.startswith("cuda"))

    best = {"workflow": -1}; history = []
    for ep in range(args.epochs):
        model.train(); tl = nb = 0; t0 = time.time()
        for xt, xu, mask, o, d in tr_ld:
            xt, xu, mask = xt.to(dev, non_blocking=True), xu.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            o, d = o.to(dev), d.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=dev.startswith("cuda")):
                out = model(xt, xu, mask)
                loss = args.organ_w * ce_o(out["organ"], o) + args.dx_w * ce_d(out["dx"], d)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            oa, da, wf, wf_ab, tau = evaluate(model, va_ld, va_cases, labels, organ_list, dx_list, tpl, dev)
        else:
            oa = da = wf = wf_ab = tau = float("nan")
        history.append(dict(epoch=ep, loss=tl/max(nb,1), organ_acc=oa, dx_acc=da,
                            workflow=wf, workflow_abstain=wf_ab, tau=tau, sec=round(time.time()-t0)))
        print(f"[{args.name}] ep{ep:03d} loss={tl/max(nb,1):.3f} organ={oa:.3f} dx={da:.3f} "
              f"wf={wf:.4f} wf_ab={wf_ab:.4f} (tau={tau}) {time.time()-t0:.0f}s", flush=True)
        if wf_ab > best["workflow"]:
            best = dict(workflow=wf_ab, workflow_noabstain=wf, organ_acc=oa, dx_acc=da,
                        epoch=ep, abstain_tau=tau)
            torch.save(lora_state_dict(model.conch), os.path.join(out_dir, "lora.pt"))
            torch.save({"config": mcfg, "state_dict": model.head.state_dict(), "abstain_tau": tau},
                       os.path.join(out_dir, "mil_head.pt"))
            json.dump({"organ": organ_list, "dx": dx_list, "dx_organ": dx_organ, "abstain_tau": tau},
                      open(os.path.join(out_dir, "label_maps.json"), "w"))
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    json.dump(best, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print(f"[{args.name}] DONE best={best}  (frozen fusion=0.814, oracle=0.889)", flush=True)


if __name__ == "__main__":
    main()
