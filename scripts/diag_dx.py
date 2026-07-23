#!/usr/bin/env python3
"""DX-bottleneck diagnostic: where does the held-out dx error mass live?

Loads a trained fusion MIL checkpoint, runs the SAME hierarchical (organ-conditioned)
inference as evaluate(), and reports:
  - overall organ/dx accuracy (sanity vs training logs)
  - per TRUE-organ dx accuracy + share of total dx-error mass
  - top confused (true_dx -> pred_dx) pairs, globally and within the worst organ
This decides whether the residual gap is CONCENTRATED (fixable: per-organ heads /
sampling) or DIFFUSE (label-noise floor -> stop).
"""
import argparse, json, os, sys
from collections import defaultdict, Counter
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np, torch

import scripts.train_mil as T
from scripts.train_mil import hash_split, SlideDS, collate
import scripts.train_fusion_mil as F
from reg2026.labels import build_label_space, dx_label_to_organ_dx, field, ORGAN_Q
from reg2026.mil import MILClassifier
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="f2_fuse_dxw")
    ap.add_argument("--v2", action="store_true", help="3-encoder (adds Virchow2)")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()
    F.USE_V2 = args.v2

    out_dir = os.path.join(F.OUT_ROOT, args.name)
    ck = torch.load(os.path.join(out_dir, "mil_head.pt"), map_location="cpu", weights_only=False)
    model = MILClassifier.from_config(ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()

    data = json.load(open(F.DATA))
    _, va = hash_split(data)
    ls = build_label_space(data)
    organ_list, dx_list, labels = ls["organ_list"], ls["dx_list"], ls["labels"]

    cache, miss, mism = F.load_fusion_cache([c["id"] for c in va])
    print(f"# held-out fused cache: {len(cache)} slides (skip {miss} missing / {mism} mismatch), "
          f"in_dim={ck['config']['in_dim']}", flush=True)

    va_ld = DataLoader(SlideDS(va, labels, cache, train=False), batch_size=32,
                       shuffle=False, collate_fn=collate, num_workers=4)
    items = va_ld.dataset.items
    cases_by_id = {c["id"]: c for c in va}

    no = nd = tot = 0
    organ_dxhit = defaultdict(lambda: [0, 0])       # true_organ -> [dx_correct, n]
    confusion = Counter()                            # (true_organ, true_dx, pred_dx) -> count for ERRORS
    samename = [0, 0]                                 # [same-disease-name errors, all errors]
    idx = 0
    for x, mask, o, d in va_ld:
        out = model(x, mask)
        po_t = out["organ"].argmax(-1)
        pd_t = model.masked_dx_logits(out["dx"], po_t).argmax(-1)
        no += (po_t == o).sum().item(); nd += (pd_t == d).sum().item(); tot += len(o)
        for k in range(len(o)):
            cid = items[idx][0]; idx += 1
            gt_organ = field(cases_by_id[cid]["chain-of-thought"], ORGAN_Q) or "?"
            td, pd = int(d[k]), int(pd_t[k])
            organ_dxhit[gt_organ][0] += int(pd == td); organ_dxhit[gt_organ][1] += 1
            if pd != td:
                _, tdx = dx_label_to_organ_dx(dx_list[td])
                _, pdx = dx_label_to_organ_dx(dx_list[pd])
                confusion[(gt_organ, tdx, pdx)] += 1
                if tdx == pdx:                       # same disease name, wrong organ-index
                    samename[0] += 1
                samename[1] += 1

    print(f"\n# OVERALL  organ_acc={no/tot:.4f}  dx_acc={nd/tot:.4f}  (n={tot})")
    total_err = tot - nd
    print(f"# of {total_err} dx-errors, {samename[0]} ({samename[0]/total_err:.1%}) have IDENTICAL "
          f"true/pred disease name (organ-alias artifact, not a real dx miss)")
    print(f"# disease-name dx_acc (counting same-name as correct) = {(nd+samename[0])/tot:.4f}")

    print(f"\n# PER-TRUE-ORGAN  (sorted by error-mass share; {total_err} total dx errors)")
    print(f"{'organ':<14}{'n':>5}{'dx_acc':>8}{'errs':>6}{'%of_errmass':>12}")
    rows = []
    for org, (hit, n) in organ_dxhit.items():
        errs = n - hit
        rows.append((org, n, hit / n if n else 0, errs, errs / total_err if total_err else 0))
    for org, n, acc, errs, share in sorted(rows, key=lambda r: -r[3]):
        print(f"{org:<14}{n:>5}{acc:>8.3f}{errs:>6}{share:>11.1%}")

    print(f"\n# TOP {args.topk} CONFUSED dx PAIRS (true -> pred), global")
    for (org, tdx, pdx), c in confusion.most_common(args.topk):
        print(f"  [{org}] {tdx!r:45} -> {pdx!r}  x{c}")

    worst = max(rows, key=lambda r: r[3])[0]
    print(f"\n# CONFUSIONS within worst-error organ = {worst!r}")
    for (org, tdx, pdx), c in confusion.most_common():
        if org == worst:
            print(f"  {tdx!r:45} -> {pdx!r}  x{c}")


if __name__ == "__main__":
    main()
