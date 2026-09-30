#!/bin/bash -l
# Draft only: requires separate user approval; afterok of train/generation.
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=omni-noise-refold
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
test "${OMNI_CONTINUATION_APPROVED:-}" = "two-gpu-hours"
BENCH="$PWD/experiments/standard_bench"
export PYTHONPATH="$BENCH/vendor/openfold:$PWD/experiments/noise_matched_poc:$PWD/experiments/native_condition_poc:$PWD/experiments/omni_joint:$PWD/experiments/official_poc"
export PATH="$BENCH/runtime/bin:$PATH"
export TORCH_HOME="$BENCH/models/torch"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
srun "$BENCH/runtime/bin/python" experiments/noise_matched_poc/refold.py --authorized-continuation
export PYTHONPATH="$PWD/experiments/noise_matched_poc:$PWD/experiments/native_condition_poc:$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint:$PWD/experiments/official_poc"
srun "$PWD/experiments/omni_joint/runtime/bin/python" experiments/noise_matched_poc/report.py
