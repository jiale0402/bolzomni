#!/bin/bash -l
set -euo pipefail
unset PYTHONPATH PYTHONHOME _WANDB_SERVICE WANDB_SERVICE WANDB_MODE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export OMNI_BENCH_CONFIG="$FUSION_ROOT/benchmark/run.json" INTERFACE_WORKERS=8
"${FUSION_SCORE_PYTHON:-python}" -u "$FUSION_ROOT/interface_launch.py"
"${FUSION_SCORE_PYTHON:-python}" -u "$FUSION_ROOT/compare_results.py"
python3 "$FUSION_ROOT/pipeline.py" --advance
