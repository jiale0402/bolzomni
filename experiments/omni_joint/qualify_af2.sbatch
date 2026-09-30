#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=omni-v2-af2-qualify
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
experiments/omni_joint/af2-runtime/bin/colabfold_batch \
  experiments/omni_joint/results/baseline-validation/designs.fasta \
  experiments/omni_joint/results/baseline-validation/af2 \
  --model-type alphafold2_ptm --num-models 1 --model-order 4 \
  --msa-mode single_sequence --num-recycle 3 --random-seed 1729 \
  --data experiments/omni_joint/af2-weights
