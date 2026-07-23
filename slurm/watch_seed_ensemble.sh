#!/bin/bash
# wait until the seed-ensemble array leaves the queue, then eval the decorrelated ensemble
until [ -z "$(squeue -u ubaid -h -n reg-seed-ens 2>/dev/null)" ]; do sleep 60; done
sleep 5
cd /group/anantm-g00/REG2026
echo "=== seed jobs done; trained runs present: ==="
ls -d artifacts/mil/f2_fuse_dxw_s* 2>/dev/null
echo "=== decorrelated ensemble eval ==="
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 /scratch/ubaid/REG2026/eval_venv/bin/python \
  scripts/eval_ensemble.py f2_fuse_dxw f1_fuse_big f2_fuse_dxw_s1 f2_fuse_dxw_s2 f2_fuse_dxw_s3 2>&1 \
  | grep -vE "NVML|FutureWarning|deserializers|warnings.warn"
