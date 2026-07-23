#!/bin/bash
#SBATCH --job-name=reg-lora
#SBATCH --partition=rp6b-8-gm768-c192-m2048,rp6b-1-gm96-c8-m64,rp6b-1-gm96-c8-m64-bk,l4-4-gm96-c48-m192
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ubaid/REG2026/logs/train_lora.out

set -e
source /group/anantm-g00/REG2026/venv/bin/activate
cd /group/anantm-g00/REG2026
echo "[lora] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
echo "[lora] tiles cached: $(ls artifacts/tiles/train/ 2>/dev/null | wc -l)"
# warm-start the MIL head from the best frozen-fusion run so LoRA only has to refine
python scripts/train_lora.py --name lora_conch_fuse --epochs 20 --k-tiles 64 --bs 4 \
  --lora-lr 1e-4 --head-lr 3e-4 --r 8 --alpha 16 --dx-w 1.5 --organ-w 0.4 \
  --warm-head f2_fuse_dxw
echo "[lora] DONE $(date)"
