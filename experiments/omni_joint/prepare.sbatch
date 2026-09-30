#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=omni-v2-data
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
/projects/p53127/envs/BoltzGenAdapter/bin/python scripts/prepare_cath.py \
  --root experiments/omni_joint --train-count 1622 --validation-count 200 \
  --test-count 100 --max-length 256 --exclude-test-manifest manifests/data.json \
  --reuse-verified-log logs/omni-v2-data-7554313.log
