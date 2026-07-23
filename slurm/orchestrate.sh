#!/bin/bash
# Full Track-2 pipeline orchestrator: wait current extraction -> recover 981 striped
# slides (both encoders) -> verify -> launch fusion training. Polls SLURM; idempotent.
cd /group/anantm-g00/REG2026
SB=/scratch/ubaid/REG2026
log(){ echo "[$(date +%H:%M:%S)] $*"; }

wait_gone(){  # $1 = comma-sep job names; block until none remain in queue
  while [ "$(squeue -u "$USER" -h -n "$1" 2>/dev/null | wc -l)" -ne 0 ]; do sleep 45; done
}
realcount(){  # $1 = enc, $2 = split -> echo number of REAL (shape>1) .npy
  venv/bin/python - "$1" "$2" <<'PY' 2>/dev/null
import numpy as np, glob, sys
enc, split = sys.argv[1], sys.argv[2]
r=sum(1 for p in glob.glob(f'artifacts/embeddings/{enc}/{split}/*.npy')
      if (lambda a: a.shape[0]>1)(np.load(p,mmap_mode='r')))
print(r)
PY
}

log "STEP1 wait for current train extraction (reg-u2h-l4,reg-u2h-rp6b)"
wait_gone "reg-u2h-l4,reg-u2h-rp6b"
log "current extraction done. conch_real=$(realcount conch train) uni2h_real=$(realcount uni2h train)"

log "STEP2 launch recovery for 981 striped slides (conch + uni2h, new tifffile reader)"
# NOTE: SLURM rejects submission from /group on this cluster -> submit from /scratch.
( cd "$SB" && sbatch cleanup_conch.sbatch && sbatch cleanup_uni2h.sbatch )
sleep 30
wait_gone "reg-recov-conch,reg-recov-u2h"
CR=$(realcount conch train); UR=$(realcount uni2h train)
log "recovery done. conch_real=$CR/11220  uni2h_real=$UR/11220"

log "STEP3 verify + launch fusion training"
if [ "$CR" -ge 11000 ] && [ "$UR" -ge 11000 ]; then
  JID=$( cd "$SB" && sbatch --parsable train_fusion.sbatch )
  log "TRAINING LAUNCHED job=$JID (configs f0/f1/f2/f3). conch=$CR uni2h=$UR"
  echo "ORCH_DONE_TRAINING_LAUNCHED job=$JID conch=$CR uni2h=$UR"
else
  log "VERIFY FAILED conch=$CR uni2h=$UR (<11000) — NOT launching training; needs attention"
  echo "ORCH_VERIFY_FAILED conch=$CR uni2h=$UR"
fi
