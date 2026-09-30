#!/bin/bash -l
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=omni-official-lock
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/experiments/omni_joint"
"$PWD/experiments/omni_joint/runtime/bin/python" experiments/official_poc/prepare_benchmark.py
