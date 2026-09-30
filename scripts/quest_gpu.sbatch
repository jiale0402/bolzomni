#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
export PYTHONPATH="$PWD/upstream/src:$PWD/scripts"
export HF_HOME="$PWD/cache/huggingface"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
BG_PY=/projects/p53127/envs/BoltzGenAdapter/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$1" in
  audit) "$BG_PY" scripts/cache_inputs.py --limit 1; "$BG_PY" scripts/train_feedback.py --audit-only ;;
  cache) "$BG_PY" scripts/cache_inputs.py ;;
  train) "$BG_PY" scripts/train_feedback.py --steps 600 ;;
  native) "$BG_PY" scripts/generate_panel.py --method native --steps 500 ;;
  feedback) "$BG_PY" scripts/generate_panel.py --method feedback --steps 500 ;;
  evalsmoke) eval-env/bin/python scripts/evaluate_panel.py --smoke ;;
  eval) eval-env/bin/python scripts/evaluate_panel.py ;;
  *) echo 'Unknown stage'; exit 2 ;;
esac
