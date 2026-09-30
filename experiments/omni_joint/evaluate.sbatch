#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=omni-joint-evaluate
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="$PWD/experiments/omni_joint/runtime/bin/python"
"$PY" -m pip install --disable-pip-version-check tmtools==0.3.0
"$PY" experiments/omni_joint/evaluate_generate.py --validation-only --limit 2 --steps 8
"$PY" experiments/omni_joint/evaluate_generate.py
if [ -s experiments/omni_joint/results/paired-test/designs.fasta ]; then
  experiments/omni_joint/af2-runtime/bin/colabfold_batch \
    experiments/omni_joint/results/paired-test/designs.fasta \
    experiments/omni_joint/results/paired-test/af2 \
    --model-type alphafold2_ptm --num-models 1 --model-order 4 \
    --msa-mode single_sequence --num-recycle 3 --random-seed 1729 \
    --data experiments/omni_joint/af2-weights
fi
"$PY" experiments/omni_joint/evaluate_report.py
