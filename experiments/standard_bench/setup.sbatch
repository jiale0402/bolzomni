#!/bin/bash -l
#SBATCH --account=p53252
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=omni-bench-setup
#SBATCH --output=/projects/p53252/jck3114/boltz_omni_poc/logs/%x-%j.log
set -euo pipefail
module load git/2.37.2
cd /projects/p53252/jck3114/boltz_omni_poc
ROOT="$PWD/experiments/standard_bench"
mkdir -p "$ROOT/vendor" "$ROOT/models/torch/hub/checkpoints"
if [ ! -d "$ROOT/vendor/MotifBench" ]; then git clone https://github.com/blt2114/MotifBench.git "$ROOT/vendor/MotifBench"; fi
git -C "$ROOT/vendor/MotifBench" checkout d5a714c6a376c701e29dfa66d42a1b13c5e339e0
if [ ! -d "$ROOT/vendor/openfold" ]; then git clone --depth 1 https://github.com/aqlaboratory/openfold.git "$ROOT/vendor/openfold"; fi
git -C "$ROOT/vendor/openfold" fetch --depth 1 origin 4b41059694619831a7db195b7e0988fc4ff3a307
git -C "$ROOT/vendor/openfold" checkout 4b41059694619831a7db195b7e0988fc4ff3a307
/projects/p53127/envs/BoltzGenAdapter/bin/python -m venv --system-site-packages "$ROOT/runtime"
PY="$ROOT/runtime/bin/python"
"$PY" -m pip install fair-esm==2.0.0 rootutils GPUtil dm-tree ml-collections hydra-joblib-launcher biotite==0.41.2 MDAnalysis==2.7.0 tmtools wandb tabulate seaborn psutil
"$PY" -m pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git@0478734ff7be75adde8d160e04872664d1c62e5f'
export PYTHONPATH="$ROOT/vendor/openfold"
export TORCH_HOME="$ROOT/models/torch"
"$PY" "$ROOT/setup_compat.py"
"$PY" - <<'PY'
import concurrent.futures, os, urllib.request
from pathlib import Path
p=Path(os.environ['TORCH_HOME'])/'hub/checkpoints'
names=['esmfold_3B_v1.pt','esm2_t36_3B_UR50D.pt','esm2_t36_3B_UR50D-contact-regression.pt']
def download(name):
    f=p/name
    if f.exists():return
    branch='regression' if 'regression' in name else 'models'
    url='https://dl.fbaipublicfiles.com/fair-esm/'+branch+'/'+name
    print('DOWNLOADING',name,flush=True)
    urllib.request.urlretrieve(url,str(f)+'.tmp')
    Path(str(f)+'.tmp').replace(f)
    print('DOWNLOADED',name,f.stat().st_size,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(download,names))
PY
"$PY" -c 'from esm.esmfold.v1.esmfold import ESMFold; print("ESMFOLD_IMPORT_OK")'
PYTHONPATH="$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint" "$PWD/experiments/omni_joint/runtime/bin/python" "$ROOT/audit_inputs.py"
cd "$ROOT/vendor/MotifBench/Scaffold-Lab"
PYTHONPATH="$ROOT/vendor/openfold:$PWD" "$PY" -c 'from motif_refolding import MotifRefolder; print("MOTIFBENCH_IMPORT_OK")'
git -C "$ROOT/vendor/openfold" rev-parse HEAD > "$ROOT/openfold_revision.txt"
"$PY" -m pip freeze > "$ROOT/evaluator_packages.txt"
printf '%s\n' SETUP_COMPLETE
