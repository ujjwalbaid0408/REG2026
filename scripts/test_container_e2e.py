#!/usr/bin/env python3
"""End-to-end container-path validation on the REAL test cohort (test_phase1/test1).

For every test slide, runs the EXACT container inference path:
    sample_tiles -> CONCH+UNI2-h live encode -> concat[CONCH|UNI2-h] -> MIL head
        -> predict_fields -> apply_template -> chain-of-thought
wrapped in the same try/except the container's interf1 uses. Records per slide:
  - ok:        no exception (or graceful fallback) and a non-empty chain emitted
  - schema_ok: every step has non-empty question & answer, last next_question == ""
  - parity_ok: live-encoded (organ,dx) prediction == saved-embeddings prediction
  - timing / peak RSS / predicted organ+dx

Shardable: --shard k --num-shards n. Writes JSONL to artifacts/e2e/<shard>.jsonl.
Goal: prove ZERO errors and ZERO schema violations across all 350 slides before rebuild.
"""
import argparse, os, sys, glob, json, time, traceback, resource
sys.path.insert(0, "/group/anantm-g00/REG2026")
sys.path.insert(0, "/group/anantm-g00/REG2026/repo_template/algorithm_submission_template")
import numpy as np
import torch

from src.common.templates import load_templates, apply_template
from src.common import predictor as P

TEST_DIR = "/group/anantm-g00/REG2026/Data/test_phase1/test1"
MODEL_DIR = "/group/anantm-g00/REG2026/repo_template/algorithm_submission_template/model"
EMB = "/group/anantm-g00/REG2026/artifacts/embeddings"
TPL = load_templates(os.path.join(MODEL_DIR, "templates_full.json"))


def schema_ok(cot):
    if not cot:
        return False
    for s in cot:
        if not str(s.get("question", "")).strip() or not str(s.get("answer", "")).strip():
            return False
    return cot[-1].get("next_question", "") == ""


def container_predict(wsi_path):
    """Mirror of interf1.predict_chain_of_thought (graceful fallback included)."""
    try:
        pred = P.predict_fields(wsi_path, MODEL_DIR, TPL)
    except Exception as e:
        print(f"[predict_fields raised] {os.path.basename(str(wsi_path))}: {e}", flush=True)
        pred = {"organ": None}
    cot = apply_template(pred, TPL)
    if not cot:
        cot = apply_template({"organ": None}, TPL)
    return pred, cot


def ref_predict_from_embeddings(cid):
    """Reference prediction from the precomputed test embeddings (the validated path).
    Returns (organ_str, dx_str) or None if embeddings missing."""
    pc = os.path.join(EMB, "conch", "test", cid + ".npy")
    pu = os.path.join(EMB, "uni2h", "test", cid + ".npy")
    if not (os.path.exists(pc) and os.path.exists(pu)):
        return None
    c = np.load(pc).astype(np.float32); u = np.load(pu).astype(np.float32)
    if c.shape[0] != u.shape[0]:
        return None
    feats = torch.from_numpy(np.concatenate([c, u], axis=1)).unsqueeze(0).to(P._STATE["device"])
    m = P._STATE["model"]; labels = P._STATE["labels"]
    with torch.no_grad():
        lo = m(feats)
        po_t = lo["organ"].argmax(-1)
        dxm = m.masked_dx_logits(lo["dx"], po_t)
        po, pd = int(po_t.item()), int(dxm.argmax(-1).item())
    from src.common.mil import dx_label_to_organ_dx
    o = labels["organ"][po] if 0 <= po < len(labels["organ"]) else None
    _, d = dx_label_to_organ_dx(labels["dx"][pd]) if 0 <= pd < len(labels["dx"]) else (None, None)
    return (o, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{TEST_DIR}/*.tiff"))
    files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    if args.limit:
        files = files[:args.limit]

    out_dir = "/group/anantm-g00/REG2026/artifacts/e2e"
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"shard{args.shard}.jsonl")
    fout = open(outp, "w")

    n_err = n_bad_schema = n_parity_mismatch = n_parity_checked = 0
    t0 = time.time()
    for k, f in enumerate(files):
        cid = os.path.basename(f).replace(".tiff", "")
        t = time.time(); err = None
        try:
            pred, cot = container_predict(f)
            ok = True
        except Exception as e:                       # should never happen (guarded), but catch
            err = f"{type(e).__name__}: {e}"; traceback.print_exc()
            pred, cot, ok = {"organ": None}, [], False
        sok = schema_ok(cot)
        # parity vs saved embeddings
        par = None
        ref = ref_predict_from_embeddings(cid) if ok else None
        if ref is not None:
            n_parity_checked += 1
            par = (ref[0] == pred.get("organ")) and (ref[1] == pred.get("dx"))
            if not par:
                n_parity_mismatch += 1
        if not ok or err:
            n_err += 1
        if not sok:
            n_bad_schema += 1
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        rec = dict(id=cid, ok=ok, schema_ok=sok, err=err, parity=par,
                   organ=pred.get("organ"), dx=pred.get("dx"), n_steps=len(cot),
                   sec=round(time.time() - t, 1), peak_gb=round(peak, 2))
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        if (k + 1) % 10 == 0:
            print(f"[shard {args.shard}] {k+1}/{len(files)} "
                  f"err={n_err} bad_schema={n_bad_schema} parity_mismatch={n_parity_mismatch}"
                  f"/{n_parity_checked} peak={peak:.1f}GB", flush=True)
    fout.close()
    print(f"[shard {args.shard}] DONE {len(files)} slides in {(time.time()-t0)/60:.1f}min | "
          f"ERRORS={n_err} BAD_SCHEMA={n_bad_schema} "
          f"PARITY_MISMATCH={n_parity_mismatch}/{n_parity_checked}", flush=True)


if __name__ == "__main__":
    main()
