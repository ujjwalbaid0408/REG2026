#!/bin/bash
#SBATCH --job-name=reg-fuse-full
#SBATCH --partition=rp6b-8-gm768-c192-m2048,l4-4-gm96-c48-m192
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ubaid/REG2026/logs/train_fusion_full.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
echo "[fuse-full] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
# config 2 = f2_fuse_dxw (best held-out: 0.8143, dx 0.737); --full = deployment model on
# ALL 11,220 slides (now with recovered striped-prostate embeddings). -> f2_fuse_dxw_full/
python scripts/train_fusion_mil.py --config 2 --full
echo "[fuse-full] FINISHED $(date)"
