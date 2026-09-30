#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=omni-bench-report
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/experiments/omni_joint"
PY="$PWD/experiments/standard_bench/runtime/bin/python"
"$PY" experiments/standard_bench/audit_results.py
"$PY" experiments/standard_bench/report.py
