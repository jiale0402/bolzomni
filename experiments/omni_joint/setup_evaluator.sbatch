#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=omni-v2-af2-setup
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
cd /projects/p53252/jck3114/boltz_omni_poc
ENV_PATH="$PWD/experiments/omni_joint/af2-runtime"
if [ ! -f "$ENV_PATH/bin/python" ]; then
  /projects/p53127/envs/BoltzGenAdapter/bin/python -m venv "$ENV_PATH"
fi
"$ENV_PATH/bin/pip" install 'colabfold[alphafold]==1.6.3' 'jax[cuda12]'
"$ENV_PATH/bin/pip" freeze > experiments/omni_joint/af2-runtime-freeze.txt
"$ENV_PATH/bin/colabfold_batch" --help > experiments/omni_joint/colabfold-help.txt
"$ENV_PATH/bin/python" - <<'PY'
from pathlib import Path
from colabfold.download import download_alphafold_params
download_alphafold_params('alphafold2_ptm',data_dir=Path('experiments/omni_joint/af2-weights'))
print('AF2_INSTALL_AND_WEIGHTS_COMPLETE; GPU smoke still required')
PY
