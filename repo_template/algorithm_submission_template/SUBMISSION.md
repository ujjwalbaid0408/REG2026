# Building and submitting the REG2026 container

The container is **code-only**; trained weights ship separately as `model.tar.gz`. The CONCH-MIL
inference path is wired in `src/common/{predictor,mil,wsi}.py`. Requires Docker (not available on
the SLURM login node — build on a workstation or a Docker-enabled node).

## 1. Assemble the model artifacts

```bash
cd repo_template/algorithm_submission_template
# copies the deployment MIL head, label maps, templates, and CONCH weights into ./model/
REPO_ROOT=/group/anantm-g00/REG2026 MIL_RUN=r1_reg_full ./prepare_model.sh
```

This produces:

```
model/
  conch/pytorch_model.bin    CONCH encoder (~802 MB)
  mil_head.pt                deployment MIL head (trained on all 11,220 slides)
  label_maps.json            organ/dx maps + dx_organ + abstain_tau
  templates_full.json        deterministic graphs/answers/reports
```

## 2. Build and smoke-test the image

```bash
./do_build.sh        # builds reg2026_algorithm (installs openslide + CONCH deps)
./do_test_run.sh     # runs the bundled debug case; verifies the I/O contract
```

`do_test_run.sh` mounts `model/` as the model path, runs both interfaces, and checks that
`chain-of-thought.json` (Interface 1) and `visual-context-response.json` (Interface 0) are
produced. Confirm a non-trivial reasoning graph is emitted (not just the baseline global
template), which indicates the MIL head loaded.

## 3. Export and upload

```bash
./do_save.sh         # saves the image tarball AND packs model.tar.gz
```

Then on Grand Challenge:
1. Upload the image `reg2026_algorithm_*.tar.gz` → **Algorithm › Container images**
2. Upload `model.tar.gz` → **Algorithm › Models** (separate from the image)
3. Run the **debug** phase first; once it passes, submit to the leaderboard.

## Notes / gotchas
- The image is fully offline (`HF_HUB_OFFLINE=1`); the encoder loads from the bundled weights,
  not Hugging Face.
- `libopenslide0` is installed in the Dockerfile (system dependency of `openslide-python`).
- The A10G target is `sm_86`; the base `cu126` build supports it (the Blackwell `cu128` upgrade
  was only needed for the *training* cluster, not deployment).
- If the MIL head fails to load for any reason, the predictor falls back to the global modal
  template so the container still returns valid output (baseline ~0.25).
- Per-case budget is dominated by tiling + one CONCH forward over ≤160 tiles — well under 5 min.
