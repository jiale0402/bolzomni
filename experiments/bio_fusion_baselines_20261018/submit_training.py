"""Submit exactly one prepared training job, with fresh account eligibility tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    config = json.loads((root/'code/config.json').read_text())
    assert config['training_enabled']
    assert not any((r/'STOPPED.json').exists() for r in [root, root.parent])
    assert not (root/'training/STARTED.json').exists(), 'Do not repeat started training'
    prepared = json.loads((root/'PREPARED.json').read_text())
    for name, digest in prepared['source_hashes'].items():
        assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==digest, name
    assert hashlib.sha256((root/'code/config.json').read_bytes()).hexdigest()==prepared['config_sha256']
    assert hashlib.sha256((root/'train.sbatch').read_bytes()).hexdigest()==prepared['train_sbatch_sha256']
    # Permanent intent lock prevents a lost SSH response from causing duplicate submissions.
    lock = root/'TRAIN_SUBMISSION_INTENT'
    lock.mkdir(exist_ok=False)
    intent = dict(utc=datetime.now(timezone.utc).isoformat(), method=config['method_id'], seed=config['seed'],
                  state='submission_attempt_started', root=str(root))
    (lock/'owner.json').write_text(json.dumps(intent, indent=2)+'\n')
    env = dict(os.environ, FUSION_ROOT=str(root))
    name = f"bio-{config['method_id']}-seed{config['seed']}-train"
    command = ['python3', str(root/'code/submit.py'), '--partition',os.environ.get('GPU_PARTITION','gpu'), '--time','04:15:00',
               '--mem','64G', '--cpus','4', '--gres','gpu:a100:1', '--constraint','pcie',
               '--name',name, str(root/'train.sbatch')]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    (lock/'stdout.txt').write_text(result.stdout)
    (lock/'stderr.txt').write_text(result.stderr)
    assert result.returncode==0, result.stdout+result.stderr+'\nInspect the intent and Slurm state before any retry.'
    audit = json.loads(result.stdout)
    job = audit['stdout'].strip().split(';')[0]
    assert job.isdigit(), audit
    receipt = dict(intent, state='submitted', job_id=job, account=audit['account'],
                   gpu='one A100 PCIe', budget_seconds=14400, slurm_time_limit='04:15:00',
                   auto_generation_or_scoring=False)
    (root/'TRAIN_SUBMITTED.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
