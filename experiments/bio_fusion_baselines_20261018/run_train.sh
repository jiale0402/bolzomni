#!/bin/bash -l
set -euo pipefail
export FUSION_BUDGET_START_UNIX="$(date +%s)"
unset PYTHONPATH PYTHONHOME _WANDB_SERVICE WANDB_SERVICE
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
: "${FUSION_ROOT:?Set FUSION_ROOT to the freshly prepared seed run}"
export FUSION_INTERFACE="$(python3 -c 'import json,os; print(json.load(open(os.environ["FUSION_ROOT"]+"/code/config.json"))["interface"])')"
export PYTHONHASHSEED="$(python3 -c 'import json,os; print(json.load(open(os.environ["FUSION_ROOT"]+"/code/config.json"))["seed"])')"
TRAIN_PYTHON="${FUSION_TRAIN_PYTHON:-python}"
exec "$TRAIN_PYTHON" -u "$FUSION_ROOT/code/train.py"
