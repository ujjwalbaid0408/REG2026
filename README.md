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

An oracle analysis bounds the workflow score at **0.889** given perfect (organ, diagnosis); our
trained model reaches **0.794** on a held-out split (organ acc 0.96, diagnosis acc 0.69). The
residual gap is fine-grained diagnosis accuracy.

## Pipeline

```
WSI ─► bounded-budget tiler ─► CONCH encoder (512-d) ─► gated-attention MIL
        (~1.4 s/slide)            (frozen)               ├─ organ head (10)
                                                         └─ diagnosis head (77, organ-conditioned)
                                                              │
                                          deterministic template engine ◄┘
                                                              │
                                 CoT graph + answers + keyword-rich report
ROI ─► Otsu tissue/background ─► visual-grounding response (Interface B)
```

## Repository layout

```
reg2026/                 core package
  canon.py               question/edge canonicalization (metric-exact strings)
  labels.py              supervised label space (organ 10-way, dx 77-way bucketed)
  templates.py           build_templates / apply_template (deterministic back end)
  metrics.py             offline workflow-score proxy for model selection
  encoder.py             CONCH / UNI2-h tile encoders (per-encoder normalization)
  mil.py                 gated-attention MIL + hierarchical dx head
scripts/
  extract_embeddings.py  WSI -> CONCH tile embeddings (sharded, resumable)
  train_mil.py           train MIL head; hierarchical eval + abstention; --full mode
  eval_mil.py            per-organ + sample-prediction evaluation
  eval_oracle.py         oracle workflow-score ceilings
slurm/                   SLURM job scripts (extraction, training)
repo_template/           offline submission container (Docker)
report/                  scientific report (LaTeX source + PDF)
requirements.txt         Python dependencies
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # installs the reg2026 package
# CONCH encoder (gated on Hugging Face — request access at huggingface.co/MahmoodLab/CONCH):
git clone https://github.com/mahmoodlab/CONCH && pip install -e CONCH
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

```bash
# 1. Oracle ceilings (templating upper bounds)
python scripts/eval_oracle.py

# 2. Extract CONCH tile embeddings (sharded; resumable)
python scripts/extract_embeddings.py --split train --shard 0 --num-shards 4 --encoder conch
#    -> artifacts/embeddings/conch/train/<id>.npy   (fp16, (<=160, 512))

# 3. Train the MIL head (sweep of 4 configs; or a single config)
python scripts/train_mil.py --config 1            # held-out 80/20, hierarchical eval + abstention
python scripts/train_mil.py --config 1 --full     # deployment model on all data

# 4. Detailed evaluation (per-organ + sample predictions)
python scripts/eval_mil.py --name r1_reg_hier
```

SLURM equivalents are in `slurm/` (set partition/account for your cluster). Key gotchas observed
on our cluster: submit from a non-`/group` filesystem; request GPUs with `--gpus=N`.

## Submission container

```bash
cd repo_template/algorithm_submission_template
# Place trained weights + CONCH snapshot + templates in the model tarball (see do_save.sh).
./do_build.sh        # build the code-only image
./do_test_run.sh     # run on the bundled debug case; checks the I/O contract
./do_save.sh         # export image + model.tar.gz for upload
```

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
| **Trained MIL (CONCH + gated attention)** | **0.794** |

See `report/main.pdf` for the full analysis, ablations, and figures.

## License / data

Code released for reproduction. Challenge data is distributed by the REG2026 organizers and is
not redistributed here.
