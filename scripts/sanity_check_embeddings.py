#!/usr/bin/env python3
"""Pre-training sanity check for extracted tile embeddings.

Validates more than counts: shape/dim, NaN/Inf, degenerate zero-vectors, value ranges,
cross-encoder tile alignment (fusion needs tile i to match), and label coverage. Run before
training (and again after the striped-slide recovery).

    python scripts/sanity_check_embeddings.py --split train
"""
import argparse, glob, json, os, sys
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np

ROOT = "/group/anantm-g00/REG2026"
EXPECT_DIM = {"conch": 512, "uni2h": 1536}


def check_encoder(enc, split, expect_total, sample=600):
    d = f"{ROOT}/artifacts/embeddings/{enc}/{split}"
    files = glob.glob(f"{d}/*.npy")
    real = ph = bad_dim = 0
    real_ids = set()
    for p in files:
        try:
            a = np.load(p, mmap_mode="r")
        except Exception:
            ph += 1; continue
        if a.ndim != 2 or a.shape[1] != EXPECT_DIM[enc]:
            bad_dim += 1; continue
        if a.shape[0] <= 1:
            ph += 1
        else:
            real += 1
            real_ids.add(os.path.basename(p)[:-4])
    # deep checks on a sample of real files (load fully)
    nan = inf = allzero = 0
    means, stds = [], []
    samp = sorted(real_ids)[:: max(1, len(real_ids) // sample)][:sample] if real_ids else []
    for cid in samp:
        a = np.load(f"{d}/{cid}.npy").astype(np.float32)
        if np.isnan(a).any(): nan += 1
        if np.isinf(a).any(): inf += 1
        if not np.any(a): allzero += 1
        means.append(float(a.mean())); stds.append(float(a.std()))
    print(f"\n[{enc}/{split}] files={len(files)} real={real} placeholder={ph} "
          f"bad_dim={bad_dim} missing={expect_total - len(files) if expect_total else '?'}")
    if samp:
        print(f"  sample({len(samp)}): NaN={nan} Inf={inf} all-zero={allzero} "
              f"| value mean={np.mean(means):+.3f} std={np.mean(stds):.3f}")
    return real_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    expect = 11220 if args.split == "train" else 350

    data = json.load(open(f"{ROOT}/Data/{'train_CoT.json' if args.split=='train' else 'train_CoT.json'}"))
    labeled = {c["id"][:-5] if c["id"].endswith(".tiff") else c["id"] for c in data}

    print(f"=== SANITY CHECK split={args.split} (expect {expect} slides) ===")
    conch = check_encoder("conch", args.split, expect)
    uni2h = check_encoder("uni2h", args.split, expect)

    both = conch & uni2h
    # tile-count alignment on the intersection
    mism = 0
    for cid in both:
        nc = np.load(f"{ROOT}/artifacts/embeddings/conch/{args.split}/{cid}.npy", mmap_mode="r").shape[0]
        nu = np.load(f"{ROOT}/artifacts/embeddings/uni2h/{args.split}/{cid}.npy", mmap_mode="r").shape[0]
        if nc != nu:
            mism += 1
    aligned = len(both) - mism

    print(f"\n=== FUSION READINESS ===")
    print(f"  real in BOTH encoders : {len(both)}")
    print(f"  tile-count mismatches : {mism}")
    print(f"  usable fused slides   : {aligned} / {expect}")
    if args.split == "train":
        print(f"  with labels           : {len(both & labeled)}")
    verdict = "READY" if aligned >= expect * 0.98 else "INCOMPLETE — run recovery before training"
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
