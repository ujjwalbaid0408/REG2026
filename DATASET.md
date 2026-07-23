# Dataset — REG2026 Reasoning-Guided WSI Report Generation

The challenge data is distributed by the REG2026 organizers and is **not redistributed** in
this repository. This document summarizes its structure and the statistics our pipeline relies
on, computed from the official `train_CoT.json` (11,220 cases).

## Splits

| Split | Cases | Notes |
|---|---|---|
| Train | **11,220** | H&E WSIs + full chain-of-thought (CoT) + organ label |
| Test (phase 1) | **350** | WSIs only; leaderboard-scored |

Our model selection uses a deterministic **80/20 hold-out** of the training set (md5-hashed slide
id → split, reproducible).

## File layout

```
Data/train/                 *.tiff whole-slide images (H&E)
Data/train_CoT.json         [{ id, organ, chain-of-thought:[{question, answer, next_question}] }]
Data/test_phase1/test1/     *.tiff (350 leaderboard slides)
```

Each WSI is a **single-level tiled (256 px) generic TIFF with no pyramid** — there is no embedded
thumbnail/downsample, which is why naive `get_thumbnail()` reads were ~78 s/slide; our bounded
tiler (`reg2026/wsi.py`) caps `read_region` calls to reach ~3 s/slide (train) / ~6 s/slide (test).

## Organ distribution (train, 7 organs)

| Organ | Cases | Share |
|---|---:|---:|
| Prostate | 2,418 | 21.6% |
| Breast | 2,213 | 19.7% |
| Colon | 1,996 | 17.8% |
| Stomach | 1,757 | 15.7% |
| Bladder | 1,079 | 9.6% |
| Lung | 945 | 8.4% |
| Cervix | 812 | 7.2% |

(Rectum/anus and other minor sites fold into the nearest organ template; the only material organ
confusion observed is rectum→colon, which is benign for templating.)

## Chain-of-thought / reasoning graph

The CoT is a path through a **highly templated reasoning graph**:

| Property | Value |
|---|---|
| Distinct canonical questions | **92** |
| Distinct directed edges | **189** |
| Chain length (steps) | min 6 · median 16 · max 41 |

> **On these counts.** Both numbers are recomputed directly from `train_CoT.json` under the
> canonicalization in `analysis/explore_cot.py` (lowercase, whitespace-collapse, strip trailing
> punctuation). Earlier revisions of this file said 186 edges and `README.md` said 191; neither was
> right. 150 steps in the corpus carry a **blank question field**, and the analysis artifacts
> (`analysis/outputs/questions.json`, `edges_global.json`) count that empty string as a node — which
> is what inflates the totals to 93 questions and 191 edges. Excluding it gives the 92 / 189 above.
> The 2-edge difference is the pair of edges whose *source* question is blank; the 11,224 blank
> `next_question` values are simply terminal steps and were never counted as edges.

Conditioning on **(organ, #1 diagnosis)** yields ~86% graph purity and ~92% answer purity — the
core insight that lets us treat the workflow as *classify (organ, diagnosis) → emit deterministic
template graph + answers + report*. See [`README.md`](README.md) and `report/main.pdf` for the
full analysis.

## Final pathology report

The terminal CoT step answers *"What is the final pathology report?"*. Reports are short, highly
structured linearizations of the predicted fields (specimen type, ordered diagnosis list, grading
sub-scores, ancillary findings):

| Report length (chars) | min | median | mean | max |
|---|---:|---:|---:|---:|
| | 31 | 76 | 89 | 321 |

Example (Breast):

```
Breast, core needle biopsy;
  1. Invasive carcinoma of no special type, grade I (Tubule formation: 2, Nuclear grade: 1, Mitoses: 1)
  2. Ductal carcinoma in situ
  3. Microcalcification
```

This near-deterministic structure is what makes the report sub-metric (0.40 keyword-Jaccard +
0.30 PubMedBERT-embedding + 0.30 ROUGE/BLEU) tractable from predicted structured fields — see
[`RESULTS.md`](RESULTS.md) for the current report-score gap and the planned report generator.
