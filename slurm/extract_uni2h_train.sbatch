#!/bin/bash
#SBATCH --job-name=reg-uni2h
#SBATCH --partition=l4-4-gm96-c48-m192,l4-4-gm96-c48-m192-bk
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=45G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/ubaid/REG2026/logs/extract_uni2h_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
# saturate the 48-core L4 node: 4 tasks/node * 12 workers = 48 cores
export TILE_WORKERS=12
export HF_HUB_OFFLINE=1
echo "[shard ${SLURM_ARRAY_TASK_ID}/8] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK} mem=${SLURM_MEM_PER_NODE} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
python scripts/extract_embeddings.py \
  --split train --shard ${SLURM_ARRAY_TASK_ID} --num-shards 8 --encoder uni2h
echo "[shard ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
