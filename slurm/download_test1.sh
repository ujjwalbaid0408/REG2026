#!/usr/bin/env bash
# Download the REG2026 test_phase1 (test1) dataset described by the manifest.
# Mirrors `aria2c -j4 -x4 -s4`: 4 files in parallel, with resume + size checks.
set -uo pipefail

MANIFEST_URL="https://d2ffc588b8gysg.cloudfront.net/manifest_test1_20260610.json"
DEST="/group/anantm-g00/REG2026/Data/test_phase1"
JOBS=4
MANIFEST="$DEST/manifest_test1_20260610.json"

mkdir -p "$DEST"
echo "[*] Fetching manifest..."
curl -fsS "$MANIFEST_URL" -o "$MANIFEST" || { echo "manifest fetch failed"; exit 1; }

TOTAL=$(jq '.files | length' "$MANIFEST")
echo "[*] $TOTAL files, $(jq -r '(.total_size_bytes/1073741824*100|round/100)' "$MANIFEST") GB total"

# Emit: url \t relative_path \t expected_size
jq -r '.files[] | "\(.url)\t\(.path)\t\(.size)"' "$MANIFEST" > "$DEST/.filelist.tsv"

export DEST
fetch_one() {
  IFS=$'\t' read -r url rel size <<<"$1"
  local out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" ]]; then
    local have
    have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [[ "$have" == "$size" ]]; then
      echo "[skip] $rel"
      return 0
    fi
  fi
  # -C - resumes a partial file; retry on transient errors.
  if curl -fsS -C - --retry 5 --retry-delay 5 "$url" -o "$out"; then
    local now
    now=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [[ "$now" == "$size" ]]; then
      echo "[ok]   $rel"
    else
      echo "[BAD]  $rel (got $now want $size)"
      return 1
    fi
  else
    echo "[FAIL] $rel"
    return 1
  fi
}
export -f fetch_one

echo "[*] Downloading with $JOBS parallel workers -> $DEST"
xargs -d '\n' -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {} < "$DEST/.filelist.tsv"

echo "[*] Done. Verifying counts..."
OK=$(find "$DEST/test1" -name 'PIT_*.tiff' | wc -l)
echo "[*] $OK / $TOTAL files present"
