# Approaches: CONCH-only and CONCH+UNI2-h Fusion

Two models were developed and submitted to REG2026 Test Phase 1. They share the entire pipeline
(bounded-budget tiler → frozen encoder(s) → gated-attention MIL → deterministic template engine →
Otsu grounding) and **differ only in the tile encoder**, which makes them a clean controlled
comparison of representation quality. All numbers below are on the deterministic 80/20 held-out
split (MD5 id-hash, seed 0; templates built on the train split only) and on the official
leaderboard.

The binding constraint throughout is **fine-grained diagnosis accuracy**. Organ recognition is
effectively solved (~0.96), the template back end is near its purity ceiling (Edge-F1 = 0.974
under oracle inputs), and the oracle workflow ceiling given perfect (organ, diagnosis) is **0.889**.
Everything we do targets diagnosis accuracy.

---

## Approach 1 — CONCH-only (baseline)

- **Encoder:** CONCH (`MahmoodLab/CONCH`, `conch_ViT-B-16`), a vision–language model producing
  **512-d** tile embeddings. Frozen. Normalized with **OpenAI-CLIP** mean/std (not ImageNet —
  CONCH's tower is CLIP-initialized).
- **Aggregator:** gated-attention MIL (Ilse et al.), input dim 512 → shared trunk → organ head
  (10-way) + diagnosis head (77-way, organ-conditioned at inference).
- **Code:** `scripts/train_mil.py`, `reg2026/mil.py`, `reg2026/encoder.py`.
- **Deployment model:** `r1_reg_full` (trained on all 11,220 slides).

### Hyper-parameter sweep (held-out)

| Run | Config | Organ acc | Dx acc | Workflow |
|---|---|---:|---:|---:|
| `r0_base` | baseline | 0.951 | 0.682 | 0.791 |
| **`r1_reg`** | **+ regularization (best)** | **0.956** | **0.691** | **0.794** |
| `r2_big` | larger head | 0.953 | 0.683 | 0.793 |
| `r3_bal` | class-balanced sampling | 0.951 | 0.646 | 0.785 |

Takeaways: regularization helps marginally; class balancing *hurts* the case-weighted metric;
capacity does not help (limit is representational, not model size). Hierarchical dx masking
(inference-only) → 0.797; confidence abstention → τ=0 (no gain).

---

## Approach 2 — CONCH + UNI2-h early fusion

- **Encoders:** CONCH (512-d, VL) **‖** UNI2-h (`MahmoodLab/UNI2-h`, ViT-H/14, DINOv2,
  **1536-d**, vision-only). Each tile is embedded by **both** frozen encoders and the two vectors
  are **concatenated into a 2048-d** per-tile feature before the same MIL head.
- **Why it works:** the two encoders are pretrained with different objectives on different data,
  so their errors are partly decorrelated. The attention head learns per-organ which representation
  to trust. Gain is concentrated entirely in **diagnosis** accuracy.
- **Index alignment (key correctness point):** the tiler (`src/common/wsi.py` `sample_tiles`) is
  fully deterministic — no randomness, fixed-seed grid shuffle — so tile *i* is byte-identical
  across encoders. CONCH and UNI2-h produce the same tile count *N* per slide (verified, 0
  mismatches), so the streams concatenate index-for-index with no registration.
- **Cost:** one extra UNI2-h forward pass per tile at inference; the model tarball ships both
  encoders (~800 MB + ~2.6 GB). Embeddings are cached per-encoder, so fusion is a zero-cost
  concatenation at training time.
- **Code:** `scripts/train_fusion_mil.py`, `reg2026/mil.py` (`in_dim=2048`), `reg2026/gradings.py`.
- **Deployment model:** `f2_fuse_dxw_full` (trained on all 11,220 slides, striped slides recovered).

### Fusion sweep (held-out)

| Run | Config | Organ acc | Dx acc | Workflow |
|---|---|---:|---:|---:|
| `f0_fuse` | base | 0.961 | 0.731 | 0.813 |
| `f1_fuse_big` | larger head | 0.957 | 0.727 | 0.812 |
| **`f2_fuse_dxw`** | **diagnosis-loss up-weighted (λ_d=1.5), best** | **0.959** | **0.737** | **0.814** |
| `f3_fuse_grade` | + categorical grading sub-heads | 0.961 | 0.729 | 0.810 |

Takeaways: dx-loss up-weighting wins (consistent with dx being the bottleneck); a larger head does
not help (representational limit again); **grading sub-heads did not help** — gradings cannot be
squeezed out of frozen features by an auxiliary loss alone.

---

## Head-to-head

| | Approach 1 (CONCH) | Approach 2 (Fusion) |
|---|---:|---:|
| Per-tile dim | 512 | 2048 |
| Held-out organ acc | 0.956 | 0.959 |
| Held-out **dx acc** | 0.691 | **0.737** |
| Held-out workflow | 0.794 | **0.814** |
| **Test Phase 1 overall** | **0.7449** | **0.7707** |
| Edge F1 / MESS / Report (test) | 0.796 / 0.727 / 0.516 | 0.820 / 0.761 / 0.564 |

The fusion improvement (held-out +0.020, leaderboard **+0.0258**) is **diagnosis-shaped**: every
diagnosis-driven component rose while the Otsu-handled grounding metrics were flat. Report Score
rose the most (+0.048) even though the report is templated — because a correct predicted diagnosis
is what the report is keyed off.

## A prerequisite data fix (benefits both, shipped with Approach 2)

981 training slides (8.7%, **877 of them prostate**) are striped-Deflate TIFFs that OpenSlide
cannot read; they had been silently embedded as zero-vectors (polluting training) and would have
produced zero features on striped *test* slides. `src/common/wsi.py` now falls back to a **bounded
random-access strip decoder** (~0.4 GB peak regardless of slide size, vs 23–26 GB for a naive full
read) that recovers them and makes the container memory-safe on gigapixel slides. See
[`ISSUES.md`](ISSUES.md).

## Next levers (by leverage on overall score)

1. **Learned report generator** — Report sub-score (weight 0.40 within A) has ceiling 0.805 given
   correct fields vs 0.564 templated; largest single lever once diagnosis is right.
2. **k-fold ensembling** of the near-tied fusion checkpoints — cheap, no new data.
3. **Encoder fine-tuning / LoRA** — capacity-insensitivity shows the limit is signal in *frozen*
   features; only fine-tuning moves it.
4. **Controlled CONCH-only + strip-decoder submission** — to separate the fusion gain from the
   striped-TIFF data fix in the V4 leaderboard delta.
