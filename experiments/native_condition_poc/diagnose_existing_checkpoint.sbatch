#!/bin/bash -l
# User authorized this single bounded diagnostic in this thread.
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --job-name=omni-cause-inspect
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/experiments/native_condition_poc:$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint:$PWD/experiments/official_poc"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$PWD/experiments/omni_joint/runtime/bin/python" experiments/native_condition_poc/diagnose_existing_checkpoint.py --authorized-diagnostic
