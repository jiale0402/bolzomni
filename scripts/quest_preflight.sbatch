#!/bin/bash
#SBATCH --job-name=boltz-poc-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=preflight-%j.log

# Submit from the project root, with account and partition supplied explicitly.
# Activate the intended Python environment before submission.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
nvidia-smi
python scripts/preflight.py
