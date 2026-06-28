#!/usr/bin/env bash
# Assemble the model/ directory that do_save.sh packs into model.tar.gz (uploaded to Grand
# Challenge as a separate Model, NOT inside the image). Contents:
#   model/conch/pytorch_model.bin   CONCH encoder weights (~766 MB, gated on HF)
#   model/uni2h/pytorch_model.bin   UNI2-h encoder weights (~2.6 GB, gated on HF) -- FUSION
#   model/mil_head.pt               trained MIL head (deployment model: all-data fusion run)
#   model/label_maps.json           organ/dx label maps + dx_organ + abstain_tau
#   model/templates_full.json       deterministic template graphs/answers/reports
#
# NOTE: This script only needs to run on the CLUSTER (it reads the trained artifacts and the
# HF weight cache). The shipped bundle ALREADY contains a fully-staged model/ -- on a Docker
# host you do NOT run this; go straight to ./do_save.sh.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- configure these for your environment ---
REPO_ROOT="${REPO_ROOT:-/group/anantm-g00/REG2026}"
MIL_RUN="${MIL_RUN:-f2_fuse_dxw_full}"   # deployment model: CONCH+UNI2-h fusion, all 11,220 slides
CONCH_BIN="${CONCH_BIN:-$(find "$HOME/.cache/huggingface" -name pytorch_model.bin -path '*onch*' 2>/dev/null | head -1)}"
UNI2H_BIN="${UNI2H_BIN:-$(find "$HOME/.cache/huggingface" -name pytorch_model.bin -path '*UNI2-h*' 2>/dev/null | head -1)}"

MODEL_DIR="${SCRIPT_DIR}/model"
mkdir -p "${MODEL_DIR}/conch" "${MODEL_DIR}/uni2h"

echo "[prepare_model] MIL run = ${MIL_RUN}"
cp "${REPO_ROOT}/artifacts/mil/${MIL_RUN}/mil_head.pt"     "${MODEL_DIR}/mil_head.pt"
cp "${REPO_ROOT}/artifacts/mil/${MIL_RUN}/label_maps.json" "${MODEL_DIR}/label_maps.json"
cp "${REPO_ROOT}/artifacts/templates_full.json"           "${MODEL_DIR}/templates_full.json"

if [ -z "${CONCH_BIN}" ] || [ ! -f "${CONCH_BIN}" ]; then
  echo "ERROR: CONCH weights not found. Set CONCH_BIN=/path/to/pytorch_model.bin" >&2
  exit 1
fi
echo "[prepare_model] CONCH weights = ${CONCH_BIN}"
cp "${CONCH_BIN}" "${MODEL_DIR}/conch/pytorch_model.bin"

if [ -z "${UNI2H_BIN}" ] || [ ! -f "${UNI2H_BIN}" ]; then
  echo "ERROR: UNI2-h weights not found. Set UNI2H_BIN=/path/to/pytorch_model.bin" >&2
  exit 1
fi
echo "[prepare_model] UNI2-h weights = ${UNI2H_BIN}"
cp "${UNI2H_BIN}" "${MODEL_DIR}/uni2h/pytorch_model.bin"

echo "[prepare_model] model/ assembled:"
du -ah "${MODEL_DIR}"
echo "[prepare_model] Next: ./do_save.sh   (builds image + packs model.tar.gz)"
