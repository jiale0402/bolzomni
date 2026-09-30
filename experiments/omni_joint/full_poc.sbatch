#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=omni-full-poc
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="$PWD/experiments/omni_joint/runtime/bin/python"
"$PY" experiments/omni_joint/train_full_poc.py --steps 1000 --seed 1729
if "$PY" -c 'import json,sys; d=json.load(open("experiments/omni_joint/runs/full-poc-seed-1729/evaluation_gate.json")); sys.exit(0 if d["full_evaluation_allowed"] else 1)'; then
  "$PY" experiments/omni_joint/evaluate_full_generate.py
  if [ -s experiments/omni_joint/results/paired-full-poc/designs.fasta ]; then
    experiments/omni_joint/af2-runtime/bin/colabfold_batch \
      experiments/omni_joint/results/paired-full-poc/designs.fasta \
      experiments/omni_joint/results/paired-full-poc/af2 \
      --model-type alphafold2_ptm --num-models 1 --model-order 4 \
      --msa-mode single_sequence --num-recycle 3 --random-seed 1729 \
      --data experiments/omni_joint/af2-weights
  fi
  "$PY" experiments/omni_joint/evaluate_full_report.py
else
  printf '%s\n' 'FULL_POC_UNSUCCESSFUL: all fixed validation designs failed endpoint validity; costly test stopped.'
fi
