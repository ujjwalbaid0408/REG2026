#!/bin/bash
#SBATCH --job-name=prost_spec
#SBATCH --partition=rp6b-8-gm768-c192-m2048
#SBATCH --account=ai-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/ubaid/REG2026/logs/prost_spec_%j.log
set -e
cd /group/anantm-g00/REG2026
/group/anantm-g00/REG2026/venv/bin/python scripts/train_prostate_specialist.py --epochs 40
