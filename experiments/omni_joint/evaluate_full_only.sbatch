#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=omni-full-eval
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%A_%a.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$PWD/experiments/omni_joint/runtime/bin/python" experiments/omni_joint/evaluate_full_generate.py --shard "$SLURM_ARRAY_TASK_ID" --shards 2
