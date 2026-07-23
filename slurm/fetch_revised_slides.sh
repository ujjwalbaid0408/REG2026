#!/usr/bin/env bash
# Fetch the 18 revised training WSIs (manifest_revised_20260527) into a staging dir,
# verifying each file's size against the manifest. 4 parallel workers, resume-capable.
set -uo pipefail
MAN=/tmp/claude-26748/manifest_slides.json
STAGE=/group/anantm-g00/REG2026/Data/train_revised_20260527
JOBS=4
mkdir -p "$STAGE"
jq -r '.files[] | "\(.url)\t\(.path | sub("^.*/";""))\t\(.size)"' "$MAN" > "$STAGE/.filelist.tsv"

export STAGE
fetch_one() {
  IFS=$'\t' read -r url name size <<<"$1"
  local out="$STAGE/$name"
  if [[ -f "$out" && "$(stat -c%s "$out" 2>/dev/null||echo 0)" == "$size" ]]; then
    echo "[skip] $name"; return 0
  fi
  if curl -fsS -C - --retry 5 --retry-delay 5 "$url" -o "$out"; then
    local now=$(stat -c%s "$out" 2>/dev/null||echo 0)
    [[ "$now" == "$size" ]] && echo "[ok]   $name ($now)" || { echo "[BAD]  $name got=$now want=$size"; return 1; }
  else echo "[FAIL] $name"; return 1; fi
}
export -f fetch_one
xargs -d '\n' -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {} < "$STAGE/.filelist.tsv"
echo "[*] staged files:"; ls -la "$STAGE"/*.tiff | wc -l
