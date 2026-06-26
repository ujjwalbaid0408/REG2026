#!/usr/bin/env python3
"""Local validation of the submission container's prediction functions, without Docker.

Checks:
  1. Interface 1 output obeys the contract schema (list of {q,a,nq} strings, terminal
     step has next_question == "") on debug + train slides.
  2. Interface 1 baseline workflow score on held-out TRAIN slides (we have GT there),
     confirming the template path works end to end.
  3. Interface 0 returns a plain non-empty string; background ROI -> rejection,
     tissue ROI -> different (informative) answer (B1/B3 sanity).
  4. The lazy WSI tiler reads real slides quickly and returns foreground tiles.
"""
import sys, os, json, time, glob
REPO = "/group/anantm-g00/REG2026/repo_template/algorithm_submission_template"
sys.path.insert(0, REPO)
sys.path.insert(0, "/group/anantm-g00/REG2026")
import numpy as np

from src.interf1.model import predict_chain_of_thought
from src.interf0.model import predict_visual_context_response
from src.common.wsi import sample_tiles
from reg2026 import metrics
from reg2026.canon import canon

TRAIN_DIR = "/group/anantm-g00/REG2026/Data/train"
DEBUG_DIR = "/group/anantm-g00/REG2026/Data/debug"


def check_schema(cot):
    assert isinstance(cot, list) and cot, "cot must be non-empty list"
    for st in cot:
        assert set(st) == {"question", "answer", "next_question"}, f"bad keys {set(st)}"
        for k in st:
            assert isinstance(st[k], str), f"{k} not str"
    assert any(st["next_question"] == "" for st in cot), "no terminal step"
    return True


def test_tiler():
    print("\n[4] WSI tiler")
    f = sorted(glob.glob(f"{TRAIN_DIR}/*.tiff"))[0]
    t0 = time.time()
    tiles = sample_tiles(f, tile=224, max_tiles=64)
    dt = time.time() - t0
    nonwhite = (tiles.mean(-1) < 220).mean()
    print(f"    {os.path.basename(f)}: tiles={tiles.shape} {dt:.1f}s "
          f"foreground_frac={nonwhite:.2f}")
    assert tiles.shape[0] > 0 and nonwhite > 0.3


def test_interf1():
    print("\n[1+2] Interface 1 (Workflow Reasoning)")
    # schema on debug slides
    for f in sorted(glob.glob(f"{DEBUG_DIR}/*.tiff"))[:3]:
        cot = predict_chain_of_thought(wsi_path=f)
        check_schema(cot)
        print(f"    schema OK: {os.path.basename(f)} ({len(cot)} steps)")
    # score on held-out train slides (we have GT)
    gt_map = {c["id"]: c["chain-of-thought"]
              for c in json.load(open("/group/anantm-g00/REG2026/Data/train_CoT.json"))}
    ids = list(gt_map)[:30]
    scores = []
    for cid in ids:
        f = os.path.join(TRAIN_DIR, cid)
        if not os.path.exists(f):
            continue
        cot = predict_chain_of_thought(wsi_path=f)
        check_schema(cot)
        scores.append(metrics.workflow_score(cot, gt_map[cid])["total"])
    print(f"    baseline workflow score on {len(scores)} train slides: "
          f"{np.mean(scores):.3f} (expected ~0.24 global-modal baseline)")


def test_interf0(tmp="/tmp/claude-26748/reg_roi"):
    print("\n[3] Interface 0 (Visual Grounding)")
    os.makedirs(tmp, exist_ok=True)
    from PIL import Image
    # background ROI: white
    bg = Image.fromarray(np.full((256, 256, 3), 245, np.uint8))
    bg.write = None
    bg_path = f"{tmp}/bg.jpeg"; bg.save(bg_path)
    # tissue ROI: a real foreground tile
    f = sorted(glob.glob(f"{TRAIN_DIR}/*.tiff"))[0]
    tiles = sample_tiles(f, tile=256, max_tiles=8)
    Image.fromarray(tiles[0]).save(f"{tmp}/tissue.jpeg")
    qpath = f"{tmp}/q.json"
    json.dump("Is meaningful tissue present in this region?", open(qpath, "w"))
    a_bg = predict_visual_context_response(question_path=qpath, roi_image_path=bg_path)
    a_ti = predict_visual_context_response(question_path=qpath, roi_image_path=f"{tmp}/tissue.jpeg")
    print(f"    background ROI -> {a_bg[:70]!r}")
    print(f"    tissue ROI     -> {a_ti[:70]!r}")
    assert isinstance(a_bg, str) and isinstance(a_ti, str) and a_bg and a_ti
    assert a_bg != a_ti, "tissue and background answers must differ (B3)"
    assert "background" in a_bg.lower() or "non-informative" in a_bg.lower()


if __name__ == "__main__":
    test_tiler()
    test_interf1()
    test_interf0()
    print("\nALL CONTRACT CHECKS PASSED ✅")
