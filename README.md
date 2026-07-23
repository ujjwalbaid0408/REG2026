# REG2026 — Reasoning-Guided WSI Report Generation

Submission to the **REG² (REG2026) Pathologist Reasoning-Guided Report Generation Challenge**
(MICCAI 2026). Given a single H&E whole-slide image (WSI), the system emits a structured
chain-of-thought (CoT) reasoning graph, the intermediate answers, and a final pathology report,
and serves a visual-grounding interface.

## Key idea

A data analysis of the 11,220-slide training corpus shows the reasoning target is **highly
templated**: only **93 canonical questions** and **191 edges** occur, and conditioning on
**(organ, #1 diagnosis)** gives ~86% graph purity and ~92% answer purity. So the task reduces to:

> **classify (organ, diagnosis) from the WSI → emit the deterministic template graph + answers + keyword-rich report.**

An oracle analysis bounds the workflow score at **0.889** given perfect (organ, diagnosis).

## Two approaches

We build and submit **two** models that differ only in the tile encoder; everything else (tiler,
MIL head, template engine, grounding) is shared. See [`docs/APPROACHES.md`](docs/APPROACHES.md).

| Approach | Encoder(s) | Held-out workflow | Held-out dx acc | **Test Phase 1 overall** | Submission |
|---|---|---:|---:|---:|---|
| 1 — CONCH-only | CONCH (512-d) | 0.794 | 0.691 | 0.7449 (top-10) | V3 (superseded) |
| **2 — CONCH+UNI2-h fusion** | CONCH ‖ UNI2-h (2048-d) | **0.814** | **0.737** | **0.7707** (top-10) | **V4 — FINAL** |

Fusion improves the binding constraint — fine-grained **diagnosis** accuracy — and therefore every
diagnosis-driven score component. The gain held up on the official leaderboard.

> ### Final submission: **V4** (Approach 2, CONCH+UNI2-h fusion) — Overall **0.7707**
>
> V4 is the submitted and final entry. Its MIL head is `artifacts/mil/f2_fuse_dxw_full/mil_head.pt`
> (committed here), and it was built with the procedure in
> [`docs/HOST_BUILD_INSTRUCTIONS_REV4.txt`](docs/HOST_BUILD_INSTRUCTIONS_REV4.txt). It is validated
> end to end over all 350 phase-1 test slides: **0 errors, 0 schema violations, 6.3 GB peak RAM**
> against a 32 GB limit.
>
> A **V5** variant also lives in this repository — a perturbation-invariant grounding answer
> (Interface 0) plus an optional prostate tumor/no-tumor specialist head, expected ≈ +0.009. **It
> was evaluated and deliberately not submitted**: the gain did not justify rebuilding and
> re-uploading a 3.3 GB container against an already-validated one. The diagnosis model is
> identical in both. To reproduce V4 exactly, pass `SPEC_RUN=none` when staging the model
> directory (see [Submission container](#submission-container)).

## Pipeline

```
WSI ─► bounded-budget tiler ─► encoder(s) ─────────────► gated-attention MIL
        (+ striped-TIFF        Approach 1: CONCH 512-d    ├─ organ head (10)
         strip decoder)        Approach 2: CONCH‖UNI2-h   └─ diagnosis head (77, organ-conditioned)
                               = 2048-d (frozen)               │
                                          deterministic template engine ◄┘
                                                              │
                                 CoT graph + answers + keyword-rich report
ROI ─► Otsu tissue/background ─► visual-grounding response (Interface B)
```

## Documentation

| Doc | Contents |
|---|---|
| **[`REPRODUCE.md`](REPRODUCE.md)** | **End-to-end reproduction of the submitted 0.7707 result** — every command, expected numbers per stage, runtimes |
| **[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)** | **Environment setup + dependency installation** (system libs, CUDA/torch matrix, encoders, report-metric env) |
| **[`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md)** | **Weights** — what ships in-tree, what to download from Hugging Face, hosting for the 3.3 GB build bundle |
| [`docs/HOST_BUILD_INSTRUCTIONS_REV4.txt`](docs/HOST_BUILD_INSTRUCTIONS_REV4.txt) | The exact build/upload procedure used for the **final submitted container** (V4) |
| [`docs/APPROACHES.md`](docs/APPROACHES.md) | **Both approaches** (CONCH-only & CONCH+UNI2-h fusion): design, configs, ablations, leaderboard |
| [`DATASET.md`](DATASET.md) | Dataset structure, splits, organ distribution, reasoning-graph + report statistics |
| [`RESULTS.md`](RESULTS.md) | Leaderboard breakdown, oracle ceilings, MIL ablations, training curves (figures) |
| [`docs/CONTAINER_BUILD.md`](docs/CONTAINER_BUILD.md) | Build/test/upload the submission container on a local Docker host |
| [`docs/ISSUES.md`](docs/ISSUES.md) | Known issues, root causes & fixes (striped-TIFF reader, placeholder masking, scheduling) |
| `report/main.pdf`, `report/REG2026_report.docx` | Full scientific report (analysis, network diagrams, confusion, qualitative) — PDF and Word |

## Repository layout

```
reg2026/                 core package
  canon.py               question/edge canonicalization (metric-exact strings)
  labels.py              supervised label space (organ 10-way, dx 77-way bucketed)
  templates.py           build_templates / apply_template (deterministic back end)
  metrics.py             offline workflow-score proxy for model selection
  encoder.py             CONCH / UNI2-h tile encoders (per-encoder normalization)
  mil.py                 gated-attention MIL + hierarchical dx head + fusion (2048-d) + grading heads
  gradings.py            categorical grading fields (Gleason/Nottingham/…) for the aux heads
  report_metric.py       faithful offline replica of the report sub-metric (BLEU/ROUGE/keyword/embed)
scripts/
  extract_embeddings.py  WSI -> CONCH/UNI2-h tile embeddings (sharded, resumable, strip-decoder)
  train_mil.py           train CONCH-only MIL head; hierarchical eval + abstention; --full mode
  train_fusion_mil.py    train CONCH+UNI2-h early-fusion MIL head (Approach 2)
  eval_mil.py            per-organ + sample-prediction evaluation
  eval_oracle.py         oracle workflow-score ceilings
  diag_report.py         report-metric diagnostic (held-out report sub-score breakdown)
slurm/                   SLURM job scripts (extraction, recovery, training, container e2e test)
repo_template/           offline submission container (Docker) — the inference code as submitted
report/                  scientific report (LaTeX source + PDF + DOCX)
artifacts/mil/<run>/     trained MIL-head weights (mil_head.pt) + metrics/history/label_maps
requirements.txt         dependencies (unpinned)
requirements-train.txt   PINNED training environment that produced the submitted model
```

## Setup

Full instructions — including the CUDA/GPU matrix and the report-metric environment — are in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md). Short version:

```bash
sudo apt-get install -y libopenslide0        # OpenSlide C library

python -m venv venv && source venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128    # see GPU note below
pip install -r requirements-train.txt        # pinned; or requirements.txt for latest
pip install -e .                             # installs the reg2026 package
# CONCH encoder (gated on Hugging Face — request access at huggingface.co/MahmoodLab/CONCH):
git clone https://github.com/mahmoodlab/CONCH repo_conch && pip install -e repo_conch
huggingface-cli login                 # token with gated-repo read
```

**GPU note.** RTX PRO 6000 / B200 are Blackwell (sm_120 / sm_100) and need
**PyTorch ≥ 2.9 built for CUDA 12.8** (`pip install torch --index-url
https://download.pytorch.org/whl/cu128`). Older `cu124` builds fail with
"no kernel image is available for execution on the device". The A10G deployment target (sm_86)
works with any recent build.

## Data layout

```
Data/train/                 *.tiff WSIs
Data/train_CoT.json         [{id, chain-of-thought:[{question,answer,next_question}], organ}]
Data/test_phase1/test1/     *.tiff (leaderboard)
```

## Reproduce

> **[`REPRODUCE.md`](REPRODUCE.md) is the authoritative, stage-by-stage guide** to reproducing the
> submitted **0.7707** result, with expected numbers and runtimes at every step. The snippet below
> is the condensed version.

```bash
# 1. Oracle ceilings (templating upper bounds)
python scripts/eval_oracle.py

# 2. Extract CONCH tile embeddings (sharded; resumable)
python scripts/extract_embeddings.py --split train --shard 0 --num-shards 4 --encoder conch
#    -> artifacts/embeddings/conch/train/<id>.npy   (fp16, (<=160, 512))

# 3a. Approach 1 — train the CONCH-only MIL head (sweep of 4 configs; or one config)
python scripts/train_mil.py --config 1            # held-out 80/20, hierarchical eval + abstention
python scripts/train_mil.py --config 1 --full     # deployment model on all data

# 3b. Approach 2 — extract UNI2-h embeddings too, then train the fusion head
python scripts/extract_embeddings.py --split train --shard 0 --num-shards 4 --encoder uni2h
# --config takes an INDEX: 0=f0_fuse 1=f1_fuse_big 2=f2_fuse_dxw 3=f3_fuse_grade
python scripts/train_fusion_mil.py --config 2            # best fusion config (held-out)
python scripts/train_fusion_mil.py --config 2 --full     # fusion deployment model (SUBMITTED)

# 4. Detailed evaluation (per-organ + sample predictions)
python scripts/eval_mil.py --name r1_reg_hier
```

> **Trained weights ship with this repository** — see [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md).
> All 23 MIL heads are committed in-tree under `artifacts/mil/<run>/mil_head.pt` (3–19 MB each),
> including **`f2_fuse_dxw_full` — the exact head inside the submitted 0.7707 container**. Nothing
> needs downloading to reproduce our numbers. The large CONCH (766 MB) and UNI2-h (2.6 GB)
> *foundation* encoders are third-party and **not** redistributed here — pull them from Hugging
> Face (gated). The 3.3 GB pre-staged Docker build bundle is hosted on
> [Google Drive](https://drive.google.com/file/d/1Y6uCULPwdolixwbRhqWFZXgRN0L0Q2sJ/view?usp=sharing)
> (MD5 `bf098e0dea840048e809b23c84471e2a`) — see `MODEL_WEIGHTS.md` §3. It is a convenience for
> offline Docker hosts and contains **no** weights that are not already in this repository.

SLURM equivalents are in `slurm/` (set partition/account for your cluster). Key gotchas observed
on our cluster: submit from a non-`/group` filesystem; request GPUs with `--gpus=N`.

## Submission container

`repo_template/algorithm_submission_template/` **is the submitted inference code** — the container
structure exactly as built for Grand Challenge (`src/interf1/` chain-of-thought, `src/interf0/`
visual grounding, `src/common/` shared tiler/encoders/templates).

```bash
cd repo_template/algorithm_submission_template
# stage head + both encoders + label maps + templates.
# SPEC_RUN=none reproduces the SUBMITTED V4 exactly (see note below).
REPO_ROOT=../.. MIL_RUN=f2_fuse_dxw_full SPEC_RUN=none ./prepare_model.sh
./do_build.sh        # build the code-only image
./do_test_run.sh     # run on the bundled debug case; checks the I/O contract
./do_save.sh         # export image + model.tar.gz for upload
```

The procedure used for the **final submission** is preserved verbatim in
[`docs/HOST_BUILD_INSTRUCTIONS_REV4.txt`](docs/HOST_BUILD_INSTRUCTIONS_REV4.txt) (bundle MD5,
validation evidence, upload steps, troubleshooting).

If the build host has no network or cluster filesystem, skip `prepare_model.sh` and download the
pre-staged 3.3 GB bundle instead —
[Google Drive](https://drive.google.com/file/d/1Y6uCULPwdolixwbRhqWFZXgRN0L0Q2sJ/view?usp=sharing),
verification steps in [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md) §3.

> **`SPEC_RUN=none` matters.** `prepare_model.sh` defaults to also staging
> `artifacts/mil/prostate_specialist/spec_head.pt`, a post-submission (V5) addition that was
> **not** part of the 0.7707 container. Pass `SPEC_RUN=none` to reproduce the submitted V4
> exactly; omit it to build the V5 variant.

The container runs **offline** (`HF_HUB_OFFLINE=1`), reads
`/input/images/whole-slide-image/<uid>.tiff`, and writes the CoT JSON; weights are mounted from a
separate `model.tar.gz`. If the trained head is absent the predictor falls back to the global
modal template, so the image is always valid.

## Results (held-out 80/20 split, 2,253 slides)

| Condition | Workflow |
|---|---|
| Global modal template (no inputs) | 0.239 |
| Oracle organ only | 0.674 |
| **Oracle organ + diagnosis (ceiling)** | **0.889** |
| Approach 1 — CONCH-only MIL | 0.794 |
| **Approach 2 — CONCH+UNI2-h fusion** | **0.814** |

See `report/main.pdf` (and `report/REG2026_report.docx`) for the full analysis, ablations, and figures.

## Leaderboard results (test phase 1)

Both submissions placed top-10. Scoring: `Overall = 0.70·A + 0.30·B`, where
`A = 0.05·BPV + 0.30·EdgeF1 + 0.25·MESS + 0.40·Report` and `B = mean(grounding metrics)`.

| Component (weight) | V3 CONCH-only | **V4 fusion (FINAL)** | Δ |
|---|---:|---:|---:|
| **Overall** | **0.7449** | **0.7707** | **+0.0258** |
| Edge F1 (0.30) | 0.7960 | 0.8203 | +0.0243 |
| MESS (0.25) | 0.7270 | 0.7611 | +0.0341 |
| Report Score (0.40) | 0.5160 | 0.5643 | +0.0483 |
| Binary Path Validity (0.05) | 0.3860 | 0.4200 | +0.0340 |
| Visual Grounding | 0.9750 | 0.9750 | 0.0000 |

Every **diagnosis-driven** component rose (Edge F1, MESS, Report, BPV) while the grounding metrics
were unchanged — the exact signature of a better (organ, diagnosis) predictor. **Report Score** rose
the most despite using the same deterministic template, because the report is keyed off the predicted
diagnosis: getting the field right is what lifts it.

**V4 is the final submission.** Diagnosis accuracy is the binding constraint on every component
except grounding, and it is saturated at ~0.74 against an oracle ceiling of 0.889: LoRA
fine-tuning, full encoder unfreezing, a third foundation encoder (Virchow2), and softmax
ensembling all landed within noise of the frozen 2048-d fusion (see
[`docs/APPROACHES.md`](docs/APPROACHES.md)). The residual errors are diffuse adjacent-grade
confusions — inter-observer label noise rather than recoverable signal. The remaining levers would
be a learned report generator (0.40, the largest single weight within `A`) and the grounding fix
staged as V5.

## License / data

Code released for reproduction. Challenge data is distributed by the REG2026 organizers and is
not redistributed here.
