#!/bin/bash -l
set -euo pipefail
unset PYTHONPATH PYTHONHOME _WANDB_SERVICE WANDB_SERVICE WANDB_MODE
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export FUSION_INTERFACE=single_conditioning_cross_attention
export PYTHONHASHSEED=42
export FUSION_BUDGET_START_UNIX="$(date +%s)"
/projects/p53127/envs/AnewOmni/bin/python -u "$FUSION_ROOT/code/train.py"
python3 "$FUSION_ROOT/pipeline.py" --advance
