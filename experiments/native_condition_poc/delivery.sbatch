#!/bin/bash -l
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=omni-native-delivery
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/experiments/native_condition_poc:$PWD/experiments/omni_joint:$PWD/experiments/official_poc"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PWD/experiments/omni_joint/runtime/bin/python" experiments/native_condition_poc/delivery_audit.py
