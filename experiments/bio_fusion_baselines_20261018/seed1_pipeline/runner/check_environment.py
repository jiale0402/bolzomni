"""Import the actual runtime surfaces used by each pipeline stage."""
import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from runtime import CONFIG, ROOT, atomic_json

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['generation', 'donor', 'score'])
args = parser.parse_args()
modules = {
    'generation': ['torch', 'torch_scatter', 'rdkit', 'Bio', 'scipy', 'numpy', 'wandb',
                   'api.pdb_dataset', 'api.helpers.gen_utils', 'models.LDM.data_utils'],
    'donor': ['torch', 'numpy', 'gemmi', 'yaml', 'wandb', 'boltzgen.model.models.boltz',
              'boltzgen.task.predict.data_from_yaml'],
    'score': ['torch', 'numpy', 'Bio', 'rdkit', 'scipy', 'pyrosetta', 'AutoDockTools',
              'openbabel', 'vina', 'wandb', 'data.bioparse.numbering'],
}[args.stage]
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, '__version__', 'imported'))
commands = {}
if args.stage == 'score':
    for name in ['plip', 'pdb2pqr30', 'obabel']:
        commands[name] = shutil.which(name)
        if not commands[name]:
            raise RuntimeError('Missing scoring command in scoring environment: ' + name)
    import AutoDockTools
    if not (Path(AutoDockTools.__path__[0]) / 'Utilities24/prepare_receptor4.py').is_file():
        raise RuntimeError('AutoDockTools prepare_receptor4.py is missing')
    sys.path.insert(0, str(Path(__file__).parent / 'scoring'))
    import evaluate_small_molecule_target
    for name in ['repo.tools.scoring', 'repo.tools.eval_atom_type', 'repo.tools.geometry.eval_stability']:
        importlib.import_module(name)
report = {'passed': True, 'stage': args.stage, 'python': sys.executable,
          'versions': versions, 'commands': commands}
atomic_json(ROOT / ('ENVIRONMENT_' + args.stage + '.json'), report)
print(json.dumps(report))
