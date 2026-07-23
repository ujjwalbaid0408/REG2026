#!/bin/bash
# wait until the LoRA job leaves the queue (done or failed)
until [ -z "$(squeue -h -j 520245 2>/dev/null)" ]; do sleep 120; done
sleep 5
echo "=== LoRA job left queue $(date) ==="
M=/group/anantm-g00/REG2026/artifacts/mil/lora_conch_fuse/metrics.json
if [ -f "$M" ]; then echo "metrics.json:"; cat "$M"; else echo "no metrics.json (likely failed)"; fi
echo "=== train_lora.out tail ==="; tail -30 /scratch/ubaid/REG2026/logs/train_lora.out 2>/dev/null
