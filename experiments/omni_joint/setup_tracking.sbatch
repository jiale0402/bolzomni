#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=omni-v2-tracking
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
RUNTIME="$PWD/experiments/omni_joint/runtime"
if [ ! -f "$RUNTIME/bin/python" ]; then
  /projects/p53127/envs/BoltzGenAdapter/bin/python -m venv --system-site-packages "$RUNTIME"
fi
"$RUNTIME/bin/pip" install wandb==0.30.0
"$RUNTIME/bin/pip" freeze > experiments/omni_joint/runtime-freeze.txt
"$RUNTIME/bin/python" experiments/omni_joint/tracking.py \
  --directory experiments/omni_joint/runs/tracking-check
