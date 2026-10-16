#!/bin/bash -l
set -euo pipefail
unset PYTHONPATH PYTHONHOME _WANDB_SERVICE WANDB_SERVICE WANDB_MODE
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export FUSION_INTERFACE=single_conditioning_cross_attention
export OMNI_BENCH_CONFIG="$FUSION_ROOT/benchmark/run.json"
export BENCH_WORKER_INDEX="$SLURM_ARRAY_TASK_ID"
/projects/p53127/envs/AnewOmni/bin/python -u "$FUSION_ROOT/generation_worker.py"
python3 "$FUSION_ROOT/pipeline.py" --advance
