#!/bin/bash
#SBATCH --job-name=reg-fuse-mil
#SBATCH --partition=rp6b-8-gm768-c192-m2048,l4-4-gm96-c48-m192
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/ubaid/REG2026/logs/train_fusion_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
echo "[fusion cfg ${SLURM_ARRAY_TASK_ID}] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
python scripts/train_fusion_mil.py --config ${SLURM_ARRAY_TASK_ID}
echo "[fusion cfg ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
