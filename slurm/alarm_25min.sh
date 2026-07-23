#!/bin/bash
# Wait ~25 min (25 x 60s), then snapshot extraction progress and exit (re-invokes the agent).
cd /group/anantm-g00/REG2026
for i in $(seq 1 25); do sleep 60; done
echo "ALARM_25MIN_ELAPSED $(date +%H:%M:%S)"
echo "uni2h=$(ls artifacts/embeddings/uni2h/train/ 2>/dev/null | wc -l)/11220"
echo "jobs_remaining=$(squeue -u "$USER" -h -n reg-u2h-l4,reg-u2h-rp6b 2>/dev/null | wc -l)"
