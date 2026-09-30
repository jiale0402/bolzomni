#!/bin/bash -l
# Draft only: requires separate user approval of the two GPU-hour budget.
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
#SBATCH --job-name=omni-noise-continue
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
test "${OMNI_CONTINUATION_APPROVED:-}" = "two-gpu-hours"
export PYTHONPATH="$PWD/experiments/noise_matched_poc:$PWD/experiments/native_condition_poc:$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint:$PWD/experiments/official_poc"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="$PWD/experiments/omni_joint/runtime/bin/python"
srun "$PY" experiments/noise_matched_poc/train_continue.py --authorized-continuation
srun "$PY" experiments/noise_matched_poc/generate_pair.py --authorized-continuation
