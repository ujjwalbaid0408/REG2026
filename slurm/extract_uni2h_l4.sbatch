#!/bin/bash
#SBATCH --job-name=reg-u2h-l4
#SBATCH --partition=l4-4-gm96-c48-m192
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=45G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/ubaid/REG2026/logs/extract_uni2h_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
export TILE_WORKERS=12
export HF_HUB_OFFLINE=1
echo "[L4 shard ${SLURM_ARRAY_TASK_ID}/6] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
python scripts/extract_embeddings.py \
  --split train --shard ${SLURM_ARRAY_TASK_ID} --num-shards 6 --encoder uni2h
echo "[L4 shard ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
