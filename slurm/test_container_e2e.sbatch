#!/bin/bash
#SBATCH --job-name=reg-e2e-test
#SBATCH --partition=l4-4-gm96-c48-m192,rp6b-8-gm768-c192-m2048
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/ubaid/REG2026/logs/e2e_test_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
export HF_HUB_OFFLINE=1
echo "[e2e $SLURM_ARRAY_TASK_ID] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) $(date)"
# Mem capped at 32G to mirror the GC A10G RAM limit -> if a slide OOMs here it would OOM on GC.
python scripts/test_container_e2e.py --shard ${SLURM_ARRAY_TASK_ID} --num-shards 4
echo "[e2e $SLURM_ARRAY_TASK_ID] FINISHED $(date)"
