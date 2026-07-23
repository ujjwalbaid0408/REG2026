#!/bin/bash
#SBATCH --job-name=reg-recov-u2h
#SBATCH --partition=l4-4-gm96-c48-m192
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=45G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/ubaid/REG2026/logs/cleanup_uni2h_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
# LOW concurrency to avoid the shared-FS contention that caused ~20% transient
# read failures at 32 workers. Resumable: only slides without a .npy are processed.
export TILE_WORKERS=4
export HF_HUB_OFFLINE=1
echo "[cleanup shard ${SLURM_ARRAY_TASK_ID}/4] node=$(hostname) $(date)"
python scripts/extract_embeddings.py \
  --split train --shard ${SLURM_ARRAY_TASK_ID} --num-shards 4 --encoder uni2h
echo "[cleanup shard ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
