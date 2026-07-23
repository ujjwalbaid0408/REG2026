# Model Weights

Nothing you need to reproduce our result is missing from this repository. **Every weight we
trained is committed in-tree.** The large files are third-party *foundation encoders* that we did
not train and are not licensed to redistribute — they are downloaded from Hugging Face.

---

## 1. Our trained weights — in this repo, no download needed

Committed under `artifacts/mil/<run>/`. Each run directory holds `mil_head.pt`,
`label_maps.json`, `metrics.json` and `history.json` (full per-epoch training curve).

| File | Size | What it is |
|---|---:|---|
| **`artifacts/mil/f2_fuse_dxw_full/mil_head.pt`** | **11 MB** | **The submitted V4 model.** CONCH‖UNI2-h 2048-d fusion MIL head, trained on all 11,220 slides. This is the head inside the container that scored **0.7707**. |
| `artifacts/mil/f2_fuse_dxw/mil_head.pt` | 11 MB | Same config, held-out 80/20 split — the run all reported held-out numbers come from (workflow 0.814, dx 0.737). |
| `artifacts/mil/r1_reg_full/mil_head.pt` | 3.8 MB | Approach 1 (CONCH-only) deployment head — the earlier V3 submission, 0.7449. |
| `artifacts/mil/prostate_specialist/spec_head.pt` | 4.6 MB | Prostate tumor/no-tumor binary specialist (staged for V5, not in the submitted V4). |
| `artifacts/mil/*/` (23 runs) | 252 MB total | Every ablation, seed, and negative result — see [`docs/APPROACHES.md`](docs/APPROACHES.md). |

Individual files are 3–19 MB, well under GitHub's limits, so no external hosting is required and
no `git-lfs` is needed.

## 2. Foundation encoders — download from Hugging Face

Frozen, never fine-tuned in the submitted model. Both are **gated**: request access on the model
page first, then log in.

| Encoder | HF repo | Size | Used by |
|---|---|---:|---|
| **CONCH** | [`MahmoodLab/CONCH`](https://huggingface.co/MahmoodLab/CONCH) | 766 MB | Both approaches |
| **UNI2-h** | [`MahmoodLab/UNI2-h`](https://huggingface.co/MahmoodLab/UNI2-h) | 2.6 GB | Approach 2 (fusion) only |

```bash
huggingface-cli login          # token with access to both gated repos

# CONCH — also needs the package, which is not on PyPI
git clone https://github.com/mahmoodlab/CONCH repo_conch
pip install -e repo_conch
huggingface-cli download MahmoodLab/CONCH pytorch_model.bin \
    --local-dir ~/.cache/huggingface/CONCH

# UNI2-h
huggingface-cli download MahmoodLab/UNI2-h pytorch_model.bin \
    --local-dir ~/.cache/huggingface/UNI2-h
```

`reg2026/encoder.py::load_encoder("conch"|"uni2h")` resolves them from the HF cache and attaches
the **correct per-encoder normalization** — CONCH uses OpenAI-CLIP statistics (std 0.269), UNI2-h
uses ImageNet (std 0.226). Using one encoder's statistics for the other is a silent accuracy leak;
we shipped that bug briefly and it is worth not repeating.

> UNI2-h publishes `.bin` only (no `.safetensors`), so `torch.load` is used for it.

## 3. Submission bundle — external hosting

The self-contained Docker build bundle used for the final submission is 3.3 GB, well over
GitHub's limits. It contains no new weights — it is `artifacts/mil/f2_fuse_dxw_full/mil_head.pt`
plus both downloaded encoders plus the container source, pre-staged so the build host needs no
network or cluster filesystem.

| Artifact | Size | MD5 |
|---|---:|---|
| `reg2026_submission_bundle_v4.tar.gz` | 3.3 GB | `bf098e0dea840048e809b23c84471e2a` |
| `model.tar.gz` (produced by `do_save.sh` step 3) | ~3.0 GB | built locally |

<!-- TODO(maintainer): upload reg2026_submission_bundle_v4.tar.gz to Drive/OneDrive/HF and
     replace the placeholder below with the real link. -->

**Download:** _<add hosting link here>_

```bash
# after downloading
md5sum reg2026_submission_bundle_v4.tar.gz    # must equal bf098e0dea840048e809b23c84471e2a
tar -xzf reg2026_submission_bundle_v4.tar.gz
cd reg2026_submission && ./do_save.sh
```

Full build/upload procedure: [`docs/HOST_BUILD_INSTRUCTIONS_REV4.txt`](docs/HOST_BUILD_INSTRUCTIONS_REV4.txt).

### You do not have to use the bundle

It is a convenience for an offline Docker host. To rebuild it from this repository:

```bash
cd repo_template/algorithm_submission_template
REPO_ROOT=../.. MIL_RUN=f2_fuse_dxw_full SPEC_RUN=none ./prepare_model.sh
./do_save.sh                               # builds image + packs model.tar.gz
```

`prepare_model.sh` copies from `artifacts/mil/$MIL_RUN/` and the HF cache, so with §1 (in-repo)
and §2 (downloaded) satisfied, the bundle is fully regenerable — nothing is unique to it.
`SPEC_RUN=none` excludes the post-submission prostate specialist head, reproducing the submitted
V4 exactly; drop it to build the V5 variant.

## 4. What the container expects at runtime

Grand Challenge mounts the model tarball separately from the code-only image. After extraction:

```
model/
  conch/pytorch_model.bin      766 MB   foundation encoder
  uni2h/pytorch_model.bin      2.6 GB   foundation encoder
  mil_head.pt                   11 MB   OUR trained head (in_dim=2048 -> fusion auto-detected)
  label_maps.json                       organ (10) + diagnosis (77) label spaces
  templates_full.json          380 KB   deterministic CoT templates, 168 (organ,dx) keys
```

`predictor.py` infers the architecture from the checkpoint: `in_dim==2048` loads both encoders,
`in_dim==512` loads CONCH alone. There is no flag to set. If `mil_head.pt` is absent or fails to
load, the predictor falls back to the global modal template — schema-valid output at ~0.25, which
is why a silent dependency failure looks like a working run.
