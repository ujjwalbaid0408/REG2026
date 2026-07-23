#!/bin/bash
#SBATCH --job-name=full_unfrz
#SBATCH --partition=rp6b-8-gm768-c192-m2048
#SBATCH --account=ai-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/ubaid/REG2026/logs/full_unfreeze_%j.log
set -e
cd /group/anantm-g00/REG2026
/group/anantm-g00/REG2026/venv/bin/python scripts/train_full_unfreeze.py \
  --name full_unfreeze --epochs 12 --warmup 1 --k-tiles 40 --bs 3 --enc-chunk 64 \
  --enc-lr 2e-5 --head-lr 3e-4 --warm-head f2_fuse_dxw
