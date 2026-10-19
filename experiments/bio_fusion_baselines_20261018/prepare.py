"""Create a fresh, immutable training run using external prepared data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent



def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['enzy', 'ses'], required=True)
    p.add_argument('--seed', type=int, choices=[1, 123, 42], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--holdout-root', type=Path, required=True)
    p.add_argument('--wandb-entity')
    p.add_argument('--wandb-project')
    p.add_argument('--run-name')
    p.add_argument('--enable-training', action='store_true', help='Explicitly enable this fresh run; does not submit a job')
    a = p.parse_args()
    root, base = a.output.resolve(), a.base.resolve()
    assert not root.exists(), 'Preserve existing runs; use a fresh output directory'
    assert not (root.parent/'STOPPED.json').exists(), 'Parent is stopped; do not remove historical stop markers'
    ledger = json.loads((HERE/'RELEASE_INDEX.json').read_text())
    for name, digest in ledger['files'].items():
        assert sha(HERE/name) == digest, 'Release changed: '+name
    config = json.loads((HERE/'configs'/a.method/f'seed{a.seed}.json').read_text())
    assert config['seed'] == a.seed and config['method_id'] == a.method
    assert config['budget_seconds'] == 14400 and not config['train_backbone']
    assert abs(config['trainable_parameters']/config['parameter_match_target']-1) <= config['parameter_match_relative_tolerance']
    for part in ['assets', 'source', 'inputs', 'features', 'manifests']:
        assert (base/part).exists(), 'Missing shared data component: '+str(base/part)
    assert sha(base/'manifests/manifest.json') == config['manifest_sha256']
    assert sha(base/'assets/anewomni_official.ckpt') == config['anew_checkpoint_sha256']
    assert (base/'source/AnewOmni/models/LDM/ldm_clean.py').exists()
    holdout = a.holdout_root.resolve()
    assert sha(holdout/'manifest.json') == '597ae47a91c17c47638541c5d97423e639221d95b325a8082e0ffcda1dd066ba'
    if a.wandb_entity:
        config['wandb_entity'] = a.wandb_entity
    if a.wandb_project:
        config['wandb_project'] = a.wandb_project
    if a.run_name:
        config['run_name'] = a.run_name
    config['training_enabled'] = a.enable_training
    root.mkdir(parents=True)
    shutil.copytree(HERE/'training_code'/a.method, root/'code')
    (root/'code/config.json').write_text(json.dumps(config, indent=2)+'\n')
    for part in ['assets', 'source', 'inputs', 'features', 'manifests']:
        (root/part).symlink_to(base/part, target_is_directory=True)
    (root/'holdout_reference').symlink_to(holdout, target_is_directory=True)
    shutil.copyfile(HERE/'run_train.sh', root/'train.sbatch')
    (root/'logs').mkdir()
    record = dict(utc=datetime.now(timezone.utc).isoformat(), method=a.method, seed=a.seed,
                  root=str(root), base=str(base), training_enabled=a.enable_training,
                  training_started=False, release_index_sha256=sha(HERE/'RELEASE_INDEX.json'),
                  source_hashes={p.name:sha(p) for p in (root/'code').glob('*.py')},
                  config_sha256=sha(root/'code/config.json'), train_sbatch_sha256=sha(root/'train.sbatch'),
                  auto_generation_or_scoring=False)
    (root/'PREPARED.json').write_text(json.dumps(record, indent=2)+'\n')
    if not a.enable_training:
        (root/'STOPPED.json').write_text(json.dumps({'reason':'Prepared only; create a new enabled run to train'})+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
