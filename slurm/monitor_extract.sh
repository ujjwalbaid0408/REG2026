#!/bin/bash
# Poll until all reg-u2h extraction array tasks leave the queue, then exit.
cd /group/anantm-g00/REG2026
for i in $(seq 1 180); do   # up to 180 min
  njobs=$(squeue -u "$USER" -h -n reg-u2h-l4,reg-u2h-rp6b 2>/dev/null | wc -l)
  ncount=$(ls artifacts/embeddings/uni2h/train/ 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] uni2h=${ncount}/11220 jobs_remaining=${njobs}"
  if [ "$njobs" -eq 0 ]; then
    echo "EXTRACTION_JOBS_DONE count=${ncount}"
    exit 0
  fi
  sleep 60
done
echo "MONITOR_TIMEOUT count=$(ls artifacts/embeddings/uni2h/train/ 2>/dev/null | wc -l)"
exit 0
