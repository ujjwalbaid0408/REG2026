#!/bin/bash
#SBATCH --job-name=reg-u2h-b200
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=6
#SBATCH --output=/scratch/ubaid/REG2026/logs/extract_uni2h_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
export TILE_WORKERS=32
export HF_HUB_OFFLINE=1
echo "[B200 shard ${SLURM_ARRAY_TASK_ID}/7] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
python scripts/extract_embeddings.py \
  --split train --shard ${SLURM_ARRAY_TASK_ID} --num-shards 7 --encoder uni2h
echo "[B200 shard ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
