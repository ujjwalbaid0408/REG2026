#!/usr/bin/env python3
"""Cache the deterministic per-slide tiles (uint8) so we can fine-tune the encoder.

Frozen-feature training reads cached embeddings; to LoRA-fine-tune CONCH we need the raw
tiles back. sample_tiles() is deterministic (fixed-seed grid shuffle), so the tiles cached
here are byte-identical to those used for the cached CONCH/UNI2-h embeddings -- tile i lines
up with embedding row i, which lets us fuse LoRA-CONCH(tiles) with the frozen UNI2-h cache.

Output: artifacts/tiles/<split>/<id>.npy  uint8 (N<=max_tiles, 224, 224, 3)
Sharded + resumable (skips ids whose .npy already exists). CPU-only (no GPU needed).
"""
import argparse, os, sys, glob, time
sys.path.insert(0, "/group/anantm-g00/REG2026")
sys.path.insert(0, "/group/anantm-g00/REG2026/repo_template/algorithm_submission_template")
import numpy as np
from multiprocessing import Pool
from src.common.wsi import sample_tiles

ROOT = "/group/anantm-g00/REG2026"
DATA_DIR = {"train": f"{ROOT}/Data/train", "test": f"{ROOT}/Data/test_phase1/test1"}


def find_slide(split, cid):
    base = cid[:-5] if cid.endswith(".tiff") else cid
    p = os.path.join(DATA_DIR[split], base + ".tiff")
    return p if os.path.exists(p) else None


def one(args):
    split, path, out, max_tiles = args
    if os.path.exists(out):
        return ("skip", os.path.basename(out))
    try:
        tiles = sample_tiles(path, tile=224, max_tiles=max_tiles)   # (N,224,224,3) uint8
        if tiles is None or len(tiles) == 0:
            return ("empty", os.path.basename(out))
        np.save(out, tiles.astype(np.uint8))
        return ("ok", f"{os.path.basename(out)}:{tiles.shape[0]}")
    except Exception as e:
        return ("fail", f"{os.path.basename(out)}:{type(e).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-tiles", type=int, default=160)   # match the embedding extraction
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    args = ap.parse_args()

    out_dir = f"{ROOT}/artifacts/tiles/{args.split}"
    os.makedirs(out_dir, exist_ok=True)

    slides = sorted(glob.glob(os.path.join(DATA_DIR[args.split], "*.tiff")))
    slides = [s for i, s in enumerate(slides) if i % args.num_shards == args.shard]
    jobs = []
    for s in slides:
        base = os.path.splitext(os.path.basename(s))[0]
        jobs.append((args.split, s, os.path.join(out_dir, base + ".npy"), args.max_tiles))
    print(f"[shard {args.shard}/{args.num_shards}] {len(jobs)} slides, procs={args.procs}", flush=True)

    t0 = time.time(); n = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
    with Pool(args.procs) as pool:
        for i, (st, msg) in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            n[st] += 1
            if st in ("fail", "empty"):
                print(f"  [{st}] {msg}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(jobs)}  ok={n['ok']} skip={n['skip']} "
                      f"fail={n['fail']} empty={n['empty']}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[shard {args.shard}] DONE {n} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
