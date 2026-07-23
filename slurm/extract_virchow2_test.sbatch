#!/bin/bash
#SBATCH --job-name=reg-v2-test
#SBATCH --partition=l4-4-gm96-c48-m192
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=45G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/ubaid/REG2026/logs/extract_virchow2_test_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
export TILE_WORKERS=8
export HF_HUB_OFFLINE=1
echo "[v2 test shard ${SLURM_ARRAY_TASK_ID}/2] node=$(hostname) $(date)"
python scripts/extract_embeddings.py \
  --split test --shard ${SLURM_ARRAY_TASK_ID} --num-shards 2 --encoder virchow2
echo "[v2 test shard ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
