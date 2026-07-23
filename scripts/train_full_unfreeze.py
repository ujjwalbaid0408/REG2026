#!/usr/bin/env python3
"""DEFINITIVE CHECK: FULL end-to-end unfreeze of the CONCH visual tower + fusion MIL head.

LoRA (partial unfreeze, attn adapters only) was flat: train loss fell but held-out dx
pinned at ~0.739 vs frozen 0.737. This unfreezes EVERY parameter of the CONCH ViT-B
visual tower (not just adapters) and trains end-to-end, to settle whether the frozen
representation -- or genuine label ambiguity -- is the dx ceiling.

Honest best-shot setup (to NOT lose to avoidable overfitting):
  - small encoder LR (default 2e-5) vs head LR (3e-4), cosine w/ warmup
  - warm-start the MIL head from the frozen-fusion model (start at wf 0.814)
  - gradient checkpointing on the ViT trunk (memory) + AMP
  - light random flip augmentation on cached tiles (regularization)
  - UNI2-h stays frozen (cached); only CONCH adapts, then concat -> 2048 -> MIL
Same hash split / templates / hierarchical dx mask / abstention sweep as everything
else, so dx-acc and workflow are directly comparable (frozen 0.737/0.814, oracle 0.889).
"""
import argparse, json, os, sys, time, random, math
sys.path.insert(0, "/group/anantm-g00/REG2026")
sys.path.insert(0, "/group/anantm-g00/REG2026/scripts")
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reg2026.labels import build_label_space
from reg2026.templates import build_templates
from reg2026.mil import MILClassifier
from reg2026.encoder import load_encoder
from train_mil import hash_split
from train_lora import TileBagDS, collate, evaluate, LoRAFusionMIL, OUT_ROOT, base_id, TILES, EMB_UNI2H


class FlipTileBagDS(TileBagDS):
    """TileBagDS + random h/v flips per tile-bag (train only) for regularization."""
    def __getitem__(self, i):
        t, u, o, d = super().__getitem__(i)
        if self.train:
            if random.random() < 0.5:
                t = torch.flip(t, dims=[1])   # vertical
            if random.random() < 0.5:
                t = torch.flip(t, dims=[2])   # horizontal
        return t, u, o, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full_unfreeze")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--k-tiles", type=int, default=48)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--enc-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--enc-chunk", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--attn-dim", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.40)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--dx-w", type=float, default=1.5)
    ap.add_argument("--organ-w", type=float, default=0.4)
    ap.add_argument("--warm-head", default="f2_fuse_dxw")
    ap.add_argument("--freeze-first", type=int, default=0, help="freeze first N transformer blocks")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    out_dir = os.path.join(OUT_ROOT, args.name); os.makedirs(out_dir, exist_ok=True)
    print(f"[{args.name}] FULL UNFREEZE  dev={dev}  args={vars(args)}", flush=True)

    data = json.load(open(__import__("train_lora").DATA))
    split_tr, split_va = hash_split(data)
    ls = build_label_space(data)
    organ_list, dx_list, labels = ls["organ_list"], ls["dx_list"], ls["labels"]
    n_organ, n_dx = len(organ_list), len(dx_list)
    organ_idx = {o: i for i, o in enumerate(organ_list)}
    dx_organ = [organ_idx[d.split("||")[0]] for d in dx_list]
    tpl = build_templates(split_tr)

    tr_ds = FlipTileBagDS(split_tr, labels, k_tiles=args.k_tiles, train=True)
    va_ds = TileBagDS(split_va, labels, train=False)
    print(f"train_ds={len(tr_ds)} val_ds={len(va_ds)}", flush=True)
    nw = min(12, max(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 2))
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, collate_fn=collate,
                       num_workers=nw, drop_last=True, persistent_workers=True, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=max(2, args.bs), shuffle=False, collate_fn=collate,
                       num_workers=nw, persistent_workers=True, pin_memory=True)

    conch, _ = load_encoder("conch", device=dev)
    # FULL unfreeze of the visual tower only (text tower is unused -> leave frozen).
    trunk = None
    for n, p in conch.named_parameters():
        p.requires_grad_("visual" in n)
    for nm, mod in conch.named_modules():
        if nm.endswith("visual.trunk"):
            trunk = mod
    if trunk is not None and hasattr(trunk, "set_grad_checkpointing"):
        try:
            trunk.set_grad_checkpointing(True); print("grad checkpointing ON (visual.trunk)", flush=True)
        except Exception as e:
            print(f"grad checkpointing unavailable ({e})", flush=True)
    # optionally freeze first N blocks for stability
    if args.freeze_first > 0 and trunk is not None and hasattr(trunk, "blocks"):
        for bi in range(min(args.freeze_first, len(trunk.blocks))):
            for p in trunk.blocks[bi].parameters():
                p.requires_grad_(False)
        print(f"froze first {args.freeze_first} blocks", flush=True)
    n_enc = sum(p.numel() for p in conch.parameters() if p.requires_grad) / 1e6
    print(f"trainable ENCODER params = {n_enc:.1f}M (full visual tower)", flush=True)

    mcfg = dict(in_dim=2048, hidden=args.hidden, attn_dim=args.attn_dim, dropout=args.dropout,
                n_organ=n_organ, n_dx=n_dx, dx_organ=dx_organ)
    head = MILClassifier.from_config(mcfg)
    if args.warm_head:
        ck = torch.load(f"{OUT_ROOT}/{args.warm_head}/mil_head.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(ck["state_dict"]); print(f"warm-started head from {args.warm_head}", flush=True)
    model = LoRAFusionMIL(conch, head, enc_chunk=args.enc_chunk).to(dev)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("conch.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.") and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.enc_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)

    def lr_factor(ep):
        if ep < args.warmup:
            return (ep + 1) / max(1, args.warmup)
        prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    ce_o = nn.CrossEntropyLoss(label_smoothing=args.smooth)
    ce_d = nn.CrossEntropyLoss(label_smoothing=args.smooth)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.startswith("cuda"))

    # baseline eval at ep -1 (frozen encoder + warm head) for a clean reference point
    oa, da, wf, wf_ab, tau = evaluate(model, va_ld, split_va, labels, organ_list, dx_list, tpl, dev)
    print(f"[{args.name}] ep-001 (frozen warm-head) organ={oa:.3f} dx={da:.4f} wf={wf:.4f} wf_ab={wf_ab:.4f}", flush=True)

    best = {"workflow": wf_ab, "dx_acc": da, "epoch": -1}; history = []
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
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1
        sched.step()
        oa, da, wf, wf_ab, tau = evaluate(model, va_ld, split_va, labels, organ_list, dx_list, tpl, dev)
        history.append(dict(epoch=ep, loss=tl/max(nb,1), organ_acc=oa, dx_acc=da,
                            workflow=wf, workflow_abstain=wf_ab, tau=tau, sec=round(time.time()-t0)))
        flag = ""
        if wf_ab > best["workflow"]:
            best = dict(workflow=wf_ab, workflow_noabstain=wf, organ_acc=oa, dx_acc=da, epoch=ep, abstain_tau=tau)
            flag = " *"
        print(f"[{args.name}] ep{ep:03d} loss={tl/max(nb,1):.3f} organ={oa:.3f} dx={da:.4f} "
              f"wf={wf:.4f} wf_ab={wf_ab:.4f} (tau={tau}) {time.time()-t0:.0f}s{flag}", flush=True)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    json.dump(best, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print(f"\n[{args.name}] DONE best={best}", flush=True)
    print(f"# REFERENCE: frozen fusion dx=0.737/wf=0.814 ; LoRA dx=0.739/wf=0.815 ; oracle=0.889", flush=True)


if __name__ == "__main__":
    main()
