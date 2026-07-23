#!/bin/bash
#SBATCH --job-name=reg-fuse-v2
#SBATCH --partition=rp6b-8-gm768-c192-m2048
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/ubaid/REG2026/logs/train_fusion_v2_%a.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
export HF_HUB_OFFLINE=1

V2DIR=artifacts/embeddings/virchow2/train
# --- count-verification guard: refuse to train on incomplete Virchow2 extraction ---
NV=$(ls $V2DIR/*.npy 2>/dev/null | wc -l)
echo "[fuse-v2 cfg ${SLURM_ARRAY_TASK_ID}] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) virchow2_train_npy=$NV $(date)"
if [ "$NV" -lt 11150 ]; then
  echo "ABORT: only $NV/11220 Virchow2 train embeddings present — extraction incomplete."
  exit 1
fi

# configs: 0=f0_fuse  1=f1_fuse_big  2=f2_fuse_dxw  (3-way 4608-d early fusion)
python scripts/train_fusion_mil.py --config ${SLURM_ARRAY_TASK_ID} --virchow2
echo "[fuse-v2 cfg ${SLURM_ARRAY_TASK_ID}] FINISHED $(date)"
