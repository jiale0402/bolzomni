"""Shared paths, immutable provenance, and local/W&B experiment logging."""
import hashlib
import json
import os
import random
import time
for _key in ("_WANDB_SERVICE", "WANDB_SERVICE"):
    os.environ.pop(_key, None)
from pathlib import Path

ROOT = Path(os.environ.get('FUSION_ROOT', '/projects/p53127/anewomni_work/pairformer_native10000_20260929'))
CODE = ROOT / 'code'
SOURCE = ROOT / 'source' / 'AnewOmni'
DATA = Path(os.environ.get('FUSION_DATA_ROOT','/projects/p53127/AnewOmni_data/loader_ready'))
ASSETS = Path('/projects/p53127/anewomni_work/boltz_adapter_20260922/assets')
CKPT = ROOT / 'assets' / 'anewomni_official.ckpt'
CONFIG = json.loads((CODE / 'config.json').read_text())

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)

def save_tensor(path, obj):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)

def seed(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)

def manifest():
    return json.loads((ROOT / 'manifests' / 'manifest.json').read_text())

def unique_rows(rows):
    return list({r['key']: r for r in rows}.values())

def load_official():
    import sys
    import torch
    sys.path.insert(0, str(SOURCE))
    checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = checkpoint.base_model if hasattr(checkpoint, 'base_model') else checkpoint
    assert hasattr(model, 'diffusion')
    return model, checkpoint if checkpoint is not model else None

class Tracker:
    def __init__(self, stage, config=None):
        import wandb
        self.started = time.monotonic()
        self.stage = stage
        self.path = ROOT / 'tracking' / (stage + '.jsonl')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        run_id = CONFIG['run_name'] + '-' + stage
        evaluation_path=CODE/'evaluation_policy.json'
        evaluation={'evaluation_policy':json.loads(evaluation_path.read_text()),
                    'evaluation_policy_sha256':sha(evaluation_path)} if evaluation_path.exists() else {}
        args = dict(entity=CONFIG['wandb_entity'], project=CONFIG['wandb_project'],
                    id=run_id, name=run_id, group=CONFIG['run_name'], job_type=stage,
                    dir=str(ROOT), config=CONFIG | evaluation | (config or {}), resume='allow')
        self.online = True
        try:
            self.run = wandb.init(**args, settings=wandb.Settings(init_timeout=45))
            self.online = getattr(self.run.settings, 'mode', 'online') not in ('offline', 'disabled')
        except Exception as e:
            print('WANDB_OFFLINE_FALLBACK', type(e).__name__, flush=True)
            wandb.finish()
            self.online = False
            self.run = wandb.init(**args, mode='offline')
        atomic_json(self.path.with_suffix('.status.json'), {
            'online': self.online, 'pending_sync': not self.online,
            'run_id': run_id, 'url': self.run.url if self.online else None,
            'local_run_dir': self.run.dir, 'slurm_job_id': os.getenv('SLURM_JOB_ID')})

    def log(self, record):
        record = dict(record)
        record['system/elapsed_seconds' if self.stage=='training-native' else 'elapsed_seconds'] = time.monotonic() - self.started
        with self.path.open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        self.run.log(record)

    def finish(self, summary=None, artifact_paths=()):
        import wandb
        self.run.summary.update(summary or {})
        self.run.summary['elapsed_seconds'] = time.monotonic() - self.started
        if artifact_paths:
            art = wandb.Artifact(CONFIG['run_name'] + '-' + self.stage, type='experiment')
            for p in artifact_paths:
                p = Path(p)
                if p.is_dir():
                    art.add_dir(str(p), name=p.name)
                else:
                    art.add_file(str(p), name=p.name)
            self.run.log_artifact(art)
        self.run.finish()
