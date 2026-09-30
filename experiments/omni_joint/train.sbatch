#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=omni-joint-train
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="$PWD/experiments/omni_joint/runtime/bin/python"
"$PY" experiments/omni_joint/cache_training.py --index-only
"$PY" experiments/omni_joint/lock_experiment.py
"$PY" experiments/omni_joint/train.py --steps 6000 --seed 1729
