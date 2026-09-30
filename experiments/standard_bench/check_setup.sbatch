#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=omni-bench-check
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
module load git/2.37.2
cd /projects/p53252/jck3114/boltz_omni_poc
ROOT="$PWD/experiments/standard_bench"
PY="$ROOT/runtime/bin/python"
"$PY" -m pip install tabulate seaborn psutil
export PYTHONPATH="$ROOT/vendor/openfold" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
"$PY" "$ROOT/setup_compat.py"
"$PY" -c 'from esm.esmfold.v1.esmfold import ESMFold; print("ESMFOLD_IMPORT_OK")'
PYTHONPATH="$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint" "$PWD/experiments/omni_joint/runtime/bin/python" "$ROOT/audit_inputs.py"
cd "$ROOT/vendor/MotifBench/Scaffold-Lab"
PYTHONPATH="$ROOT/vendor/openfold:$PWD" "$PY" -c 'from motif_refolding import MotifRefolder; print("MOTIFBENCH_IMPORT_OK")'
git -C "$ROOT/vendor/openfold" rev-parse HEAD > "$ROOT/openfold_revision.txt"
"$PY" -m pip freeze > "$ROOT/evaluator_packages.txt"
printf '%s\n' SETUP_COMPLETE
