# Known Issues, Root Causes & Fixes

Engineering issues found and resolved while building the CONCH+UNI2-h fusion pipeline
(2026-06-27/28). Ordered by severity. Several are **pre-existing** and affected the original
0.7449 leaderboard submission.

---

## 1. OpenSlide cannot read striped-Deflate TIFFs — 981 slides (8.7%) silently lost  🔴 critical, pre-existing

**Symptom.** ~981/11,220 training slides failed with
`OpenSlideUnsupportedFormatError: Unsupported or missing image file`. They had **zero-vector
placeholder embeddings in both CONCH and UNI2-h**.

**Root cause.** The corpus mixes two TIFF encodings:
- Most slides: **tiled** JPEG generic-TIFFs → OpenSlide reads them fine.
- ~8.7%: **striped** (`tiled=False`), **Adobe-Deflate** compressed (`compression=8`) TIFFs.
  OpenSlide's generic-TIFF support requires *tiled* images and rejects striped ones outright.

The files are **not corrupt** — `tifffile` and `PIL` read them perfectly
(e.g. shape `(17773, 30905, 3)`, uint8).

**Impact (this is the important part).** The bug predates this work:
- The **0.7449 model trained on ~981 zero-vector slides** (8.7% label-noise pollution).
- The **submitted container cannot read striped *test* slides** → zero features → wrong
  organ/dx → wrong template → **points lost on ~8.7% of the leaderboard test set**.

**Fix.** `src/common/wsi.py :: sample_tiles` now tries OpenSlide first and **falls back to
`tifffile.imread`** (full-image read) for any slide OpenSlide can't open, then slices tiles
from the array with the *same* grid/tissue logic (`_grid_sample`, `_tissue_frac`), padding
edge tiles. Verified: striped slides now yield real tiles (e.g. 87/26/44), normal slides go
through OpenSlide unchanged (no regression), 5–12 s/slide. `tifffile` is already in the
container `requirements.txt`, so **rebuilding the submission auto-fixes test inference.**

**Recovery.** The 981 placeholders are re-extracted on both encoders with the new reader
(`cleanup_conch.sbatch`, `cleanup_uni2h.sbatch`, low concurrency for the full-image memory).

---

## 2. Read-failure placeholders masked as "done" — failures never retried  🔴 critical

**Symptom.** Failed slides were never re-processed on resumable re-runs, and silently
appeared in the dataset as 1-tile zero embeddings.

**Root cause.** Two compounding bugs in `scripts/extract_embeddings.py`:
1. On a tile-read failure it wrote a placeholder `np.zeros((1, dim))` `.npy`.
2. Resumability (`todo`) only checked **file existence** — so a placeholder counted as
   "done" and the slide was never retried.

**Fix.**
- **Retry** `sample_tiles` 3× with backoff before giving up (handles genuinely transient I/O).
- **No placeholder on failure** — leave the file absent so a resumable pass retries it.
- **Placeholder-aware resumability** — `_is_done()` reads the array header (mmap) and treats
  any existing `shape[0] <= 1` file as *not* done, so the ~700 placeholders already on disk
  auto-redo.

---

## 3. Misdiagnosis: "filesystem contention" theory was wrong  🟡 process note

**What happened.** The failures first *looked* like shared-FS contention: they correlated
with worker count (32-worker RTX6000 shards failed ~20%, 12-worker L4 shards ~0%). We reduced
workers and added retries — which **did not fix it**.

**Correction.** An isolated **single-process** read of a "failed" slide still threw
`OpenSlideUnsupportedFormatError` → not contention at all (Issue #1). The worker-count
correlation was an artifact: the restarted shards redo previously-failed (striped) slides
first, so they *appear* to fail in clusters. **Lesson:** reproduce in isolation before
trusting a load-correlation.

---

## 4. NaN in grading multi-task loss  🟠 correctness

**Symptom.** Grading auxiliary loss became `NaN`, corrupting the whole training step.

**Root cause.** Grading fields are sparse (only relevant organs have them). When a batch
contained **zero valid targets** for a field, `CrossEntropyLoss(ignore_index=-1)` averaged
over 0 elements → `NaN`.

**Fix.** Add a field's loss only when it has ≥1 valid target in the batch:
`if (tgt != -1).any(): gl = gl + ce_grade(...)`. Verified finite with partial targets and
`0.0` when a batch has no graded cases.

---

## 5. SLURM scheduling constraints  🟢 operational

- **Only one L4 node schedulable.** The sibling partition `l4-4-gm96-c48-m192-bk` is
  `State=DRAIN` (admin-disabled), so only 4 L4 GPUs are usable, not 8. An 8-way array left
  half the shards `PENDING (Resources)`; rebalanced to a 6-way split (4 L4 + 2 RTX6000).
- **B200 effectively unavailable.** Our fairshare priority was **23** (bottom) behind 4
  higher-priority queued jobs; the node was 7/8 used by others; SLURM's own start estimate was
  **+12 h**. Dropped the B200 shard rather than block the pipeline.
- **Memory request too large for L4.** Training first requested `mem=200G`, which can't fit on
  L4 (≈45 G/GPU). The embedding RAM-cache is only ~15 GB, so it was lowered to 40 G and the
  job targets both RTX6000 + L4 partitions so it runs wherever GPUs free first.
- **Cluster conventions:** submit with `--gpus=N` (not `--gres=gpu:`); submit from `/scratch`,
  not `/group`; L4 jobs default to account `general`, rp6b/b200 to `ai-gpu`.

---

## 6. Fusion validity: tile alignment across encoders  🟢 verified assumption

Early fusion (per-tile concat of CONCH 512-d + UNI2-h 1536-d → 2048-d) requires that tile *i*
is the same patch in both encoders. `sample_tiles` is **deterministic** (seeded shuffle, no
unguarded randomness), so the two encoders extract identical tiles in identical order.
Verified: matching tile counts per slide across encoders, 0 mismatches. The fusion cache also
guards with a tile-count check and skips any slide where the two disagree.
