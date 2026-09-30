#!/bin/bash -l
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=omni-native-refold
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
BENCH="$PWD/experiments/standard_bench"
export PYTHONPATH="$BENCH/vendor/openfold:$PWD/experiments/native_condition_poc:$PWD/experiments/omni_joint:$PWD/experiments/official_poc"
export PATH="$BENCH/runtime/bin:$PATH"
export TORCH_HOME="$BENCH/models/torch"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
"$BENCH/runtime/bin/python" experiments/native_condition_poc/refold.py
