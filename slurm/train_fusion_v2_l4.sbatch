#!/bin/bash
#SBATCH --job-name=reg-fuse-v2-l4
#SBATCH --partition=l4-4-gm96-c48-m192-bk
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=45G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/ubaid/REG2026/logs/train_fusion_v2_l4_s%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
export HF_HUB_OFFLINE=1

V2DIR=artifacts/embeddings/virchow2/train
NV=$(ls $V2DIR/*.npy 2>/dev/null | wc -l)
echo "[fuse-v2-l4 seed ${SLURM_ARRAY_TASK_ID}] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) virchow2_train_npy=$NV $(date)"
if [ "$NV" -lt 11150 ]; then echo "ABORT: only $NV Virchow2 train embeddings."; exit 1; fi

# config 2 = f2_fuse_dxw (dx-weighted, best at 2-way); seed = array index.
# seed 0 -> f2_fuse_dxw_v2 ; seeds 1-3 -> f2_fuse_dxw_v2_s{1,2,3} (decorrelated ensemble)
python scripts/train_fusion_mil.py --config 2 --virchow2 --seed ${SLURM_ARRAY_TASK_ID}
echo "[fuse-v2-l4 seed ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
