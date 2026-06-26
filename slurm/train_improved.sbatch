#!/bin/bash
#SBATCH --job-name=reg-mil-imp
#SBATCH --partition=rp6b-8-gm768-c192-m2048
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/ubaid/REG2026/logs/train_imp_%a.out
set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
export PYTHONHASHSEED=0
echo "[imp ${SLURM_ARRAY_TASK_ID}] $(hostname) $(date)"
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  python scripts/train_mil.py --config 1            # hierarchical+abstention on 80/20 split
else
  python scripts/train_mil.py --config 1 --full     # deployment model on all data
fi
echo "[imp ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
