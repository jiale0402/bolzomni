#!/bin/bash -l
set -euo pipefail
unset PYTHONPATH PYTHONHOME _WANDB_SERVICE WANDB_SERVICE WANDB_MODE
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export FUSION_INTERFACE=enzy_author_operator_transfer
export PYTHONHASHSEED="$(python3 -c 'import json,os;print(json.load(open(os.environ["FUSION_ROOT"]+"/code/config.json"))["seed"])')"
export FUSION_BUDGET_START_UNIX="$(date +%s)"
"${FUSION_TRAIN_PYTHON:-python}" -u "$FUSION_ROOT/code/train.py"
python3 "$FUSION_ROOT/pipeline.py" --advance
