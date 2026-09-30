#!/bin/bash -l
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=omni-native-flow
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/experiments/native_condition_poc:$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint:$PWD/experiments/official_poc"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="$PWD/experiments/omni_joint/runtime/bin/python"
"$PY" experiments/native_condition_poc/train.py
"$PY" experiments/native_condition_poc/generate.py
