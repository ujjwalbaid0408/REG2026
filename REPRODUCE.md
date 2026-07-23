# Reproducing the Submitted Result

End-to-end path from the raw challenge data to the exact container that scored **0.7707** on
REG2026 test phase 1 (Approach 2, CONCH+UNI2-h fusion).

Every command below is the real one we ran. SLURM equivalents are in [`slurm/`](slurm/); adjust
`--partition` / `--account` for your site.

**Prerequisites:** [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) (environment + encoders),
[`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md) (weights).

---

## Shortcut: skip training entirely

The trained head we submitted is committed to this repository. To rebuild the submitted
container without re-running any training:

```bash
cd repo_template/algorithm_submission_template
REPO_ROOT=../.. MIL_RUN=f2_fuse_dxw_full SPEC_RUN=none ./prepare_model.sh
./do_save.sh                               # -> image tarball + model.tar.gz
```

Jump to [Stage 5](#stage-5--build-and-submit-the-container). The stages below reproduce that head
from scratch.

---

## Data layout

```
Data/train/                 11,220 *.tiff WSIs
Data/train_CoT.json         [{id, chain-of-thought:[{question,answer,next_question}], organ}]
Data/test_phase1/test1/     350 *.tiff (leaderboard phase 1)
```

Data is distributed by the REG2026 organizers and is not redistributed here. Set `REG_DATA_ROOT`
if your layout differs.

## Stage 0 — Establish the ceiling (~2 min, CPU)

```bash
python scripts/eval_oracle.py
```

| Condition | Workflow |
|---|---:|
| Global modal template (no input) | 0.239 |
| Oracle organ only | 0.674 |
| **Oracle organ + diagnosis** | **0.889** |

This is what justifies the whole design: given `(organ, diagnosis)`, everything downstream is a
deterministic template lookup. **0.889 is the ceiling** — the modeling problem is a two-head
classification problem, not a generation problem.

## Stage 1 — Extract tile features (~10 h on 4 GPUs, the expensive stage)

Both encoders over all 11,220 training slides, then the 350 test slides. Sharded and resumable —
re-running skips completed slides.

```bash
# CONCH (512-d)
python scripts/extract_embeddings.py --split train --encoder conch --shard 0 --num-shards 8
# UNI2-h (1536-d)
python scripts/extract_embeddings.py --split train --encoder uni2h --shard 0 --num-shards 8
# test split, both encoders
python scripts/extract_embeddings.py --split test --encoder conch --shard 0 --num-shards 2
python scripts/extract_embeddings.py --split test --encoder uni2h --shard 0 --num-shards 2
```

```bash
# SLURM: 8-way arrays
sbatch slurm/extract_train.sbatch          # CONCH
sbatch slurm/extract_uni2h_train.sbatch    # UNI2-h
sbatch slurm/extract_conch_test.sbatch
sbatch slurm/extract_uni2h_test.sbatch
```

Output: `artifacts/embeddings/<encoder>/<split>/<id>.npy`, fp16, shape `(<=160, dim)`.

Three things that matter here:

- **`TILE_WORKERS<=12`.** Higher values on a shared filesystem produce transient
  `Unsupported or missing image file` errors on perfectly good slides.
- **Tiling is deterministic**, which is what makes early fusion valid: tile *i* from the CONCH
  pass and tile *i* from the UNI2-h pass are the same pixels. Verified zero mismatches across the
  corpus. Do not introduce randomness into `sample_tiles`.
- **Alignment is per-slide.** Any change to the tiler invalidates both feature sets.

### Stage 1b — Verify, and recover the striped slides

```bash
python scripts/sanity_check_embeddings.py
```

~981 slides (8.7%, **almost all prostate**) are striped Adobe-Deflate TIFFs that OpenSlide cannot
open. `src/common/wsi.py` falls back to a bounded band-major strip decoder for them. On an older
extractor these wrote `(1, dim)` all-zero *placeholders* that looked "done" to a
file-existence resume check — so they trained as zero vectors. The fix is in-tree, but if you are
resuming an old run:

```bash
sbatch slurm/cleanup_conch.sbatch      # re-does placeholders (shape[0]<=1) with the strip decoder
sbatch slurm/cleanup_uni2h.sbatch
sbatch slurm/recover_striped.sbatch
```

Count real features with `shape[0] > 1`, not files on disk. Expect **11,220 train / 350 test**,
minus 4 genuinely near-empty control scans (`PIT_01_00515/07777/07800/09119`) that legitimately
yield one tile. This recovery is worth real points — it was crippling the largest organ class.

## Stage 2 — Train the fusion MIL head (~2 h, 1 GPU)

Per-tile `CONCH(512) ‖ UNI2-h(1536) = 2048-d` early fusion into a gated-attention MIL head with
organ (10-way) and organ-conditioned diagnosis (77-way) outputs.

```bash
# held-out 80/20 (deterministic md5 split) — the reported numbers
python scripts/train_fusion_mil.py --config 2        # config 2 = f2_fuse_dxw

# all four configs
sbatch slurm/train_fusion.sbatch                     # array 0-3
```

`--config` takes an **index**, not a name: `0=f0_fuse`, `1=f1_fuse_big`, `2=f2_fuse_dxw`,
`3=f3_fuse_grade`.

Expected held-out results (2,253 slides):

| Config | Workflow | Organ | Diagnosis |
|---|---:|---:|---:|
| `f0_fuse` | 0.8130 | 0.961 | 0.731 |
| `f1_fuse_big` | 0.8121 | — | — |
| **`f2_fuse_dxw`** | **0.8143** | **0.959** | **0.737** |
| `f3_fuse_grade` | 0.8103 | — | — |
| *Approach 1 (CONCH-only, `r1_reg`)* | *0.794* | *0.956* | *0.691* |

Diagnosis weighting (`f2`) wins; auxiliary grading heads (`f3`) do **not** help. The entire gain
over Approach 1 is diagnosis accuracy, +0.046.

## Stage 3 — Train the deployment head on all data

```bash
python scripts/train_fusion_mil.py --config 2 --full     # -> artifacts/mil/f2_fuse_dxw_full/
sbatch slurm/train_fusion_full.sbatch
```

**This produces the submitted model.** `--full` trains on all 11,220 slides, so its reported
validation score is leaked by construction (val ⊂ train) — ignore it. Model selection was done in
Stage 2 on the clean split; this stage only refits the chosen configuration on more data.

## Stage 4 — Evaluate

```bash
python scripts/eval_mil.py --name f2_fuse_dxw       # per-organ breakdown + sample predictions
python scripts/diag_dx.py                           # where diagnosis errors concentrate
python scripts/whatif_roi.py                        # sizes each error bucket in workflow-score terms
python scripts/diag_report.py                       # report sub-metric (needs eval_venv, §4 of ENVIRONMENT.md)
```

`whatif_roi.py` is the one worth running if you plan to improve on this: it scores hypothetical
fixes in *workflow-score* terms rather than raw accuracy, which is what revealed that the
remaining diagnosis errors are diffuse adjacent-grade confusions (well↔moderately differentiated,
grade II↔III, LSIL↔HSIL) rather than a fixable subgroup.

## Stage 5 — Build and submit the container

```bash
cd repo_template/algorithm_submission_template
REPO_ROOT=../.. MIL_RUN=f2_fuse_dxw_full SPEC_RUN=none ./prepare_model.sh
./do_test_run.sh                          # I/O contract smoke test on the bundled case
./do_save.sh                              # image tarball + model.tar.gz
```

`prepare_model.sh` reads the trained head from `artifacts/mil/$MIL_RUN/` and both encoders from
the Hugging Face cache (override with `CONCH_BIN=` / `UNI2H_BIN=` if they live elsewhere).

> **`SPEC_RUN=none` is required to reproduce V4.** The script defaults to also staging
> `artifacts/mil/prostate_specialist/spec_head.pt` — a post-submission (V5) prostate
> tumor/no-tumor override head that was **not** in the 0.7707 container. Leaving the default in
> place builds the V5 variant, not the submitted one.

**Confirm the model actually loaded** before submitting:

```
[predictor] fusion model loaded (CONCH+UNI2-h -> 2048)
```

If that line is missing, a dependency failed and inference silently fell back to the template
baseline (~0.25) while still emitting valid output. This is the single most likely way to waste a
submission. See `docs/HOST_BUILD_INSTRUCTIONS_REV4.txt` STEP 1a.

Upload **both** artifacts to Grand Challenge — the image is code-only, and without the matching
`model.tar.gz` the container has no weights:

1. *Container Images* → `reg2026_algorithm_<timestamp>.tar.gz`
2. *Models* → `model.tar.gz`

Full procedure: [`docs/HOST_BUILD_INSTRUCTIONS_REV4.txt`](docs/HOST_BUILD_INSTRUCTIONS_REV4.txt).
Container internals: [`docs/CONTAINER_BUILD.md`](docs/CONTAINER_BUILD.md).

### Pre-submission validation

```bash
python scripts/local_contract_test.py                 # I/O contract + output schema
sbatch slurm/test_container_e2e.sbatch                # all 350 test slides through the real path
```

Our run: **0 errors, 0 schema violations, 6.3 GB peak RAM** (Grand Challenge allows 32 GB),
comfortably inside the 5 min/case budget.

---

## Expected final result

| | Held-out workflow | Test phase 1 Overall |
|---|---:|---:|
| Approach 1 — CONCH-only | 0.794 | 0.7449 |
| **Approach 2 — CONCH+UNI2-h fusion** | **0.814** | **0.7707** |
| Oracle ceiling | 0.889 | — |

`Overall = 0.70·A + 0.30·B`, `A = 0.05·BPV + 0.30·EdgeF1 + 0.25·MESS + 0.40·Report`,
`B = mean(grounding metrics)`. Component breakdown in [`RESULTS.md`](RESULTS.md).

## Runtime and hardware

| Stage | Hardware | Wall clock |
|---|---|---|
| 1 — feature extraction (both encoders, 11,570 slides) | 4× L4 / 2× RTX PRO 6000 | ~10 h |
| 2 — fusion training (4 configs) | 1 GPU | ~2 h each |
| 3 — deployment head | 1 GPU | ~2.5 h |
| 5 — container build | Docker host, no GPU | ~20 min |

Stage 1 dominates and is the only stage worth parallelizing. Everything after it operates on
cached features and is cheap — which is also why the frozen-encoder design was the right call
under the challenge timeline.
