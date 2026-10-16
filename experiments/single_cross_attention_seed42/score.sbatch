#!/bin/bash -l
set -euo pipefail
unset PYTHONPATH PYTHONHOME _WANDB_SERVICE WANDB_SERVICE WANDB_MODE
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export FUSION_INTERFACE=single_conditioning_cross_attention
export OMNI_BENCH_CONFIG="$FUSION_ROOT/benchmark/run.json"
export BENCH_SCORE_WORKERS=24 INTERFACE_WORKERS=8
/projects/p53127/envs/AnewOmniEval/bin/python -u "$FUSION_ROOT/score_launch.py"

python3 "$FUSION_ROOT/pipeline.py" --advance
