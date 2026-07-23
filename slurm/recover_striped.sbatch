#!/bin/bash
#SBATCH --job-name=reg-recov13
#SBATCH --partition=l4-4-gm96-c48-m192,rp6b-8-gm768-c192-m2048
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/ubaid/REG2026/logs/recover_striped_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
ENCS=(conch uni2h)
ENC=${ENCS[$SLURM_ARRAY_TASK_ID]}
export TILE_WORKERS=4          # bounded strip reader; sparse slides do an exhaustive scan
export HF_HUB_OFFLINE=1
echo "[recover13 $ENC] node=$(hostname) $(date)"
# num-shards 1 scans all train slides; only the ~13 remaining placeholders (shape<=1)
# are not-done, so it reprocesses exactly those via the new bounded wsi.py.
python scripts/extract_embeddings.py \
  --split train --shard 0 --num-shards 1 --encoder $ENC --max-tiles 160
echo "[recover13 $ENC] FINISHED $(date)"
