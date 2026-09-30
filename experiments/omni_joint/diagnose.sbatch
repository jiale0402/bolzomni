#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --job-name=omni-diagnose
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
experiments/omni_joint/runtime/bin/python experiments/omni_joint/diagnose_failure.py
