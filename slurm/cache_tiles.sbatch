#!/bin/bash
#SBATCH --job-name=reg-cache-tiles
#SBATCH --partition=c64-m512,c128-m1024,oxf-c64-m512
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/ubaid/REG2026/logs/cache_tiles_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
echo "[cache tiles shard ${SLURM_ARRAY_TASK_ID}/8] node=$(hostname) $(date)"
python scripts/cache_tiles.py --split train --shard ${SLURM_ARRAY_TASK_ID} --num-shards 8 --max-tiles 160 --procs 30
echo "[cache tiles shard ${SLURM_ARRAY_TASK_ID}] DONE $(date)"
