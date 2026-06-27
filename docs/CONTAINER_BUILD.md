# Submission Container — Build, Test & Upload (Docker host / local machine)

> Step-by-step instructions for building the REG2026 algorithm container on a local Docker host and
> uploading it to [reg2026.grand-challenge.org](https://reg2026.grand-challenge.org). The container
> is **code-only**; trained weights ship **separately** as `model.tar.gz`.

**Current revision: REV 3** — bakes in three source-level fixes (no manual edits needed):

1. **Empty-question schema fix** — the original submission blocker (see below).
2. `requirements.txt` now includes **`transformers==4.46.3`**. The vendored CONCH hard-imports
   `transformers` at load time; without it the model load is silently swallowed and inference falls
   back to the baseline (~0.25).
3. `do_save.sh` STEP 3 tar line is now **quoted** (`"$output_tarball_name"`), so it no longer breaks
   when the build folder path contains spaces.

### Why the first debug submission failed (and how it was fixed)

Grand Challenge rejected the output with:

```
The output file 'chain-of-thought.json' is not valid.
JSON does not fulfill schema: instance ... should be non-empty
```

**Cause:** 5 Lung `(organ, diagnosis)` templates contained a reasoning step whose `question` field
was an empty string `""`; GC's output schema requires `question`/`answer` to be non-empty.

**Fixed two ways:** (a) repaired the empty questions in `templates_full.json` (in both the staged
`model/` and `src/common/`), and (b) added a defensive guard in `src/common/templates.py` so the
container can never emit an empty `question`/`answer` again. Verified: all 179 possible emitted
chains now have 0 empty required fields.

> You MUST rebuild the image **and** re-upload **both** artifacts (image + `model.tar.gz`) — an old
> container on GC would still carry the bug.

---

## What this is

A self-contained REG2026 algorithm submission. The bundle already has `model/` fully staged (CONCH
encoder + MIL head + label maps + templates), so you do **not** need the cluster filesystem or any
network / Hugging Face access — everything is inside the tarball.

**Target inference environment (Grand Challenge):** single **A10G GPU (sm_86)**, ≤32 GB RAM,
**<5 min/case**, fully **offline**.

## Prerequisites on the build machine

- Docker installed and running (`docker --version`)
- ~5 GB free disk (image + tarballs)
- *(Optional)* NVIDIA GPU + `nvidia-container-toolkit` — not required to build or to run the contract
  smoke-test on CPU, but recommended to mirror the real run.

---

## Step 0 — Verify the transfer

```bash
md5sum reg2026_submission_bundle_v3.tar.gz
# expect: 08cf17ba99cd3653c4eeb51549279ce3
# (macOS:  md5 reg2026_submission_bundle_v3.tar.gz)
```

## Step 1 — Extract

```bash
tar -xzf reg2026_submission_bundle_v3.tar.gz
cd reg2026_submission
chmod +x do_*.sh prepare_model.sh
```

Extracted layout:

```
reg2026_submission/
  Dockerfile            base: pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
  core.py, inference.py entrypoint + interface dispatch
  src/                  interf0 (visual grounding) + interf1 (workflow CoT)
                        + common/{predictor,mil,wsi,canon,templates}.py
  conch/                vendored CONCH package (open_clip_custom, downstream)
  requirements.txt      openslide-python, open_clip, timm, transformers, ...
  do_build.sh / do_test_run.sh / do_save.sh
  prepare_model.sh      (NOT needed here — model/ already staged)
  model/                <-- ALREADY POPULATED, do not re-run prepare_model.sh
    conch/pytorch_model.bin   CONCH encoder  (~766 MB)
    mil_head.pt               deployment MIL head (all 11,220 slides)
    label_maps.json           organ/dx maps + dx_organ + abstain_tau
    templates_full.json       deterministic graphs/answers/reports
```

> **Do NOT run `prepare_model.sh` on this machine.** Its paths point at the Emory cluster
> (`/group/...` and the HF cache) which don't exist here. `model/` is already filled.

## Step 2 — Build the image

```bash
./do_build.sh    # builds image tagged 'reg2026_algorithm'
```

Installs `libopenslide0` + python deps; first build may take several minutes. The Dockerfile pins
`--platform=linux/amd64` — on Apple Silicon this builds amd64 via emulation (slower but correct; GC
runs amd64).

## Step 3 — Smoke-test the I/O contract (the critical check)

```bash
./do_test_run.sh
```

Mounts `model/` and runs the bundled debug case through **both** interfaces:

- **Interface 1 (workflow):** `test/input/interf1/images/whole-slide-image/<uid>.tiff` →
  `chain-of-thought.json` (bare JSON array)
- **Interface 0 (grounding):** ROI jpeg + `visual-context-question.json` → plain JSON string

**Pass criteria:**

- ✅ Both output files are produced without error.
- ✅ `chain-of-thought.json` is a **non-trivial** reasoning graph (multiple question/next_question
  edges specific to the predicted organ), **not** the tiny global-modal baseline template. *This is
  the signal that CONCH + the MIL head actually loaded inside the container.* A minimal/identical
  baseline graph means the model load failed silently → the container would score ~0.25.
- ✅ **No step has an empty `question`/`answer`** — the exact condition that failed the previous
  submission:

  ```bash
  python3 -c "import json; d=json.load(open('test/output/interf1/chain-of-thought.json')); \
  bad=[i for i,s in enumerate(d) if not (s.get('question') or '').strip() or not (s.get('answer') or '').strip()]; \
  print('EMPTY FIELDS at',bad) if bad else print('OK: no empty question/answer')"
  ```

## Step 4 — Export image + model tarball

```bash
./do_save.sh
```

Produces two upload artifacts:

- `reg2026_algorithm_*.tar.gz` — the container image (**code only**)
- `model.tar.gz` — the weights (CONCH + MIL head + maps + templates)

## Step 5 — Upload to Grand Challenge

1. Upload the image `reg2026_algorithm_*.tar.gz` → **Algorithm › Container images**
2. Upload `model.tar.gz` → **Algorithm › Models** (*separate* from the image)
3. Run the **debug phase** first; verify it completes and returns valid output.
4. Only after debug passes, submit to the leaderboard.

## Expected performance (sanity-check the leaderboard number)

- Oracle ceiling (organ + #1 diagnosis): **0.889**
- Realistic held-out workflow score: **~0.794** (organ acc ~0.96, dx acc ~0.69, frozen CONCH)
- The deployment model's own reported **0.845 is leaked** (its val set is a subset of its training
  set) — do not expect 0.845 on the leaderboard.
- If you see **~0.25**, the model did not load (baseline fallback) — see Step 3 / Troubleshooting.

> The actual test-phase result for the REV-3 container was **0.7449** (top-10). See
> [`../RESULTS.md`](../RESULTS.md) for the full metric breakdown.

## Troubleshooting

- **"baseline graph only / score ~0.25"** — the MIL head or CONCH encoder didn't load. Check
  `do_test_run.sh` logs for import errors (`conch`, `open_clip`, `openslide`) or a missing `model/`
  mount. Confirm `model/conch/pytorch_model.bin` is ~766 MB (not an LFS pointer / truncated copy).
- **openslide errors** — `libopenslide0` is installed in the Dockerfile; if missing, WSI tiling
  fails. Rebuild cleanly (`docker build --no-cache`).
- **CUDA "no kernel image"** — only relevant to the Blackwell training cluster. The deployment target
  is A10G sm_86, supported by the cu126 base. On a non-NVIDIA local box the smoke-test runs on CPU —
  fine for the contract check.

## Quick reference

```bash
md5sum reg2026_submission_bundle_v3.tar.gz   # 08cf17ba99cd3653c4eeb51549279ce3
tar -xzf reg2026_submission_bundle_v3.tar.gz
cd reg2026_submission && chmod +x do_*.sh
./do_build.sh
./do_test_run.sh        # confirm NON-trivial CoT graph
./do_save.sh            # -> image tar + model.tar.gz
# then upload image -> Container images, model.tar.gz -> Models, run debug phase
```
