#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=omni-motifbench-score
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
ROOT="$PWD/experiments/standard_bench"
export PYTHONPATH="$ROOT/vendor/openfold:$PWD/experiments/omni_joint"
export PATH="$ROOT/runtime/bin:$PATH"
export TORCH_HOME="$ROOT/models/torch"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
"$ROOT/runtime/bin/python" "$ROOT/refold.py"
"$ROOT/runtime/bin/python" "$ROOT/report.py"
