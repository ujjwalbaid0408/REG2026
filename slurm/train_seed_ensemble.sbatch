#!/bin/bash
#SBATCH --job-name=reg-seed-ens
#SBATCH --partition=rp6b-1-gm96-c8-m64-bk,rp6b-1-gm96-c8-m64,l4-4-gm96-c48-m192,rp6b-8-gm768-c192-m2048
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-3
#SBATCH --output=/scratch/ubaid/REG2026/logs/seed_ens_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
echo "[seed ${SLURM_ARRAY_TASK_ID}] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
# config 2 = f2_fuse_dxw (best fusion config); vary only the seed -> decorrelated members
python scripts/train_fusion_mil.py --config 2 --seed ${SLURM_ARRAY_TASK_ID}
echo "[seed ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
