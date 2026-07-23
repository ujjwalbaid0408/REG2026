#!/bin/bash
#SBATCH --job-name=reg-v2-train
#SBATCH --partition=rp6b-8-gm768-c192-m2048
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/ubaid/REG2026/logs/extract_virchow2_train_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
export TILE_WORKERS=32          # 2 shards x 32 = 64 readers; retries+strip-reader handle transient FS fails
export HF_HUB_OFFLINE=1
echo "[v2 train shard ${SLURM_ARRAY_TASK_ID}/2] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
python scripts/extract_embeddings.py \
  --split train --shard ${SLURM_ARRAY_TASK_ID} --num-shards 2 --encoder virchow2
echo "[v2 train shard ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
