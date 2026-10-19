"""Lock validation-selected weights before any generation benchmark is inspected."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    complete=json.loads((ROOT/'training/COMPLETE.json').read_text())
    train=json.loads((ROOT/'code/config.json').read_text())
    assert complete['base_unchanged'] and not complete['budget_exceeded']
    assert isinstance(train['seed'],int) and train['budget_seconds']==14400
    assert sha(ROOT/'training/best_joint.pt')==complete['checkpoint_sha256']
    assert complete['gpu'].startswith('NVIDIA A100'),complete['gpu']
    cfg=json.loads((ROOT/'reference_benchmark_config.local.json').read_text())
    assert cfg['samples']==30 and cfg['batch_size']==10 and cfg['seed']==20260814
    assert sum(x['targets'] for x in cfg['benchmark_specs'])==253
    for spec in cfg['benchmark_specs']:assert sha(spec['manifest'])==spec['manifest_sha256']
    cfg.update(checkpoint=str(ROOT/'training/best_joint.pt'),checkpoint_sha256=complete['checkpoint_sha256'],
        output=str(ROOT/'benchmark'),projection_width=512,interface=train['interface'],
        cache_donor_projections=False,train_backbone=False,gamma=1.0,
        training_seed=train['seed'],fusion_kwargs=train['fusion_kwargs'],trainable_parameters=train['trainable_parameters'],reference_training_seed=20261007,
        training_seconds=complete['job_elapsed_seconds_before_bookkeeping'],
        source_hashes={str(p.relative_to(ROOT/'runner')):sha(p) for p in (ROOT/'runner').rglob('*.py')},
        execution_source_hashes={name:sha(ROOT/name) for name in ['generation_worker.py','score_resume.py','score_native_launch.py','compare_results.py','interface_launch.py']})
    cfg.pop('cache_equivalence_gate',None)
    cfg.pop('run_hash',None)
    cfg['run_hash']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    dest=ROOT/'benchmark/run.json'
    if dest.exists():assert json.loads(dest.read_text())==cfg,'Never change a started benchmark'
    else:
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'checkpoint_sha256':cfg['checkpoint_sha256'],'run_hash':cfg['run_hash']}))


if __name__=='__main__':main()
