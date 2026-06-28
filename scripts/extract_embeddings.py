#!/usr/bin/env python3
"""Precompute UNI2-h tile embeddings for REG2026 slides.

For each slide: sample foreground tiles -> UNI2-h -> save (N, 1536) fp16 to
artifacts/embeddings/<id>.npy. Resumable (skips existing). Shardable for parallel runs:
  python extract_embeddings.py --split train --shard 0 --num-shards 4
A CPU prefetch thread overlaps tiling with GPU embedding.
"""
import argparse, os, sys, glob, time, json, threading, queue
sys.path.insert(0, "/group/anantm-g00/REG2026")
sys.path.insert(0, "/group/anantm-g00/REG2026/repo_template/algorithm_submission_template")
import numpy as np
import torch

from reg2026.encoder import load_encoder, embed_tiles
from src.common.wsi import sample_tiles

TRAIN_DIR = "/group/anantm-g00/REG2026/Data/train"
TEST_DIR = "/group/anantm-g00/REG2026/Data/test_phase1/test1"
OUT_ROOT = "/group/anantm-g00/REG2026/artifacts/embeddings"


def slide_list(split):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return sorted(glob.glob(f"{d}/*.tiff"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-tiles", type=int, default=160)
    ap.add_argument("--encoder", default="uni2h")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N slides")
    args = ap.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.encoder, args.split)
    os.makedirs(out_dir, exist_ok=True)
    files = slide_list(args.split)
    files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    if args.limit:
        files = files[:args.limit]
    def _is_done(f):
        # A slide counts as done only if its .npy exists AND holds a REAL embedding
        # (>1 tile). Placeholder zeros (shape[0]==1) from past read-failures are redone.
        p = os.path.join(out_dir, os.path.basename(f).replace(".tiff", ".npy"))
        if not os.path.exists(p):
            return False
        try:
            return np.load(p, mmap_mode="r").shape[0] > 1
        except Exception:
            return False
    todo = [f for f in files if not _is_done(f)]
    print(f"[shard {args.shard}/{args.num_shards}] split={args.split} "
          f"assigned={len(files)} todo={len(todo)}", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc, dim = load_encoder(args.encoder, device=device)
    print(f"encoder {args.encoder} dim={dim} device={device}", flush=True)

    # CPU prefetch: a pool of tiling threads (openslide releases the GIL during JPEG
    # decode) feeds a queue; GPU embedding happens in the main thread.
    n_workers = int(os.environ.get("TILE_WORKERS", "6"))
    fq = queue.Queue()
    for f in todo:
        fq.put(f)
    out_q = queue.Queue(maxsize=2 * n_workers)
    done_workers = [0]
    lock = threading.Lock()

    def worker():
        while True:
            try:
                f = fq.get_nowait()
            except queue.Empty:
                break
            tiles, err = None, None
            for attempt in range(3):                      # retry transient FS/read errors
                try:
                    tiles = sample_tiles(f, tile=224, max_tiles=args.max_tiles)
                    break
                except Exception as e:
                    err = e
                    time.sleep(0.5 * (attempt + 1))
            if tiles is None:
                print(f"[tile-fail] {os.path.basename(f)}: {err} (after 3 tries)", flush=True)
            out_q.put((f, tiles))
        with lock:
            done_workers[0] += 1
            if done_workers[0] == n_workers:
                out_q.put(None)

    for _ in range(n_workers):
        threading.Thread(target=worker, daemon=True).start()

    n, t0 = 0, time.time()
    while True:
        item = out_q.get()
        if item is None:
            break
        f, tiles = item
        cid = os.path.basename(f).replace(".tiff", "")
        outp = os.path.join(out_dir, cid + ".npy")
        if tiles is None or len(tiles) == 0:
            # Do NOT write a placeholder: leave the file absent so a resumable re-run
            # retries this slide (placeholders previously masked ~700 read-failures as done).
            continue
        emb = embed_tiles(enc, tiles, device=device, batch=128, fp16=True)
        np.save(outp, emb.astype(np.float16))
        n += 1
        if n % 25 == 0:
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(todo)} done {rate:.2f} slide/s "
                  f"eta {((len(todo)-n)/rate/60):.0f}min", flush=True)
    print(f"[shard {args.shard}] DONE {n} slides in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
