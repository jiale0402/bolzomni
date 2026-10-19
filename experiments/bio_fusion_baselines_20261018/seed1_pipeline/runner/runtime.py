"""Run-local configuration; no cluster-specific paths or training completion gates."""
import hashlib
import json
import os
from pathlib import Path
import random
import time

HERE = Path(__file__).resolve().parent
CONFIG = json.loads(Path(os.environ['OMNI_BENCH_CONFIG']).read_text())
ROOT = Path(CONFIG['output'])
SOURCE = Path(CONFIG['runtime']['anew_source'])
ASSETS = Path(CONFIG['runtime']['boltzgen_assets'])
CACHE_ROOT = Path(CONFIG['cache'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_tensor(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def seed(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def load_model():
    import sys
    sys.path.insert(0, str(SOURCE))
    import torch
    if CONFIG.get('cache_donor_projections', False):
        from cached_fusion import Fusion, FusionEpsilon
    else:
        from fusion import Fusion, FusionEpsilon
    path = Path(CONFIG['runtime']['base_checkpoint'])
    assert sha(path) == CONFIG['base_checkpoint_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = saved.base_model if hasattr(saved, 'base_model') else saved
    assert hasattr(model, 'diffusion') and model.diffusion.num_steps == 100
    model.eval().requires_grad_(False)
    if CONFIG['variant'] == 'base':
        assert sha(CONFIG['checkpoint']) == CONFIG['base_checkpoint_sha256']
        return model, None
    checkpoint = torch.load(CONFIG['checkpoint'], map_location='cpu', weights_only=False)
    expected_joint = CONFIG['train_backbone']
    assert ('epsilon_base' in checkpoint) == expected_joint
    if expected_joint:
        model.diffusion.eps_net.load_state_dict(checkpoint['epsilon_base'], strict=True)
    fusion = Fusion(CONFIG['variant'], CONFIG['projection_width'], **CONFIG['fusion_kwargs'])
    assert sum(p.numel() for p in fusion.parameters() if p.requires_grad) == CONFIG['trainable_parameters']
    fusion.load_state_dict(checkpoint['fusion'], strict=True)
    fusion.eval().requires_grad_(False)
    wrapper = FusionEpsilon(model.diffusion.eps_net, fusion, gamma=CONFIG['gamma'])
    model.diffusion.eps_net = wrapper
    return model, wrapper


class Tracker:
    def __init__(self, stage, config=None):
        import wandb
        self.started = time.monotonic()
        self.path = ROOT / 'tracking' / (stage + '.jsonl')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        run_id = 'bench-' + CONFIG['run_hash'][:12] + '-' + stage
        kwargs = dict(project=CONFIG['wandb_project'], entity=CONFIG.get('wandb_entity'),
                      id=run_id, name=run_id, group=CONFIG['run_hash'][:12], job_type=stage,
                      dir=str(ROOT), config=CONFIG | (config or {}), resume='allow')
        try:
            self.run = wandb.init(**kwargs, settings=wandb.Settings(init_timeout=30))
        except Exception as exc:
            print('WANDB_OFFLINE_FALLBACK', type(exc).__name__, flush=True)
            wandb.finish()
            self.run = wandb.init(**kwargs, mode='offline')
        online = self.run.settings.mode == 'online'
        atomic_json(self.path.with_suffix('.status.json'), {
            'online': online, 'pending_sync': not online, 'run_id': run_id,
            'url': self.run.url if online else None, 'local_run_dir': self.run.dir,
            'slurm_job_id': os.getenv('SLURM_JOB_ID')})

    def log(self, values):
        values = dict(values, elapsed_seconds=time.monotonic() - self.started)
        with self.path.open('a') as handle:
            handle.write(json.dumps(values, allow_nan=False) + '\n')
        self.run.log(values)

    def finish(self, summary=None, artifact_paths=()):
        import wandb
        self.run.summary.update(summary or {})
        if artifact_paths:
            artifact = wandb.Artifact('benchmark-' + self.run.id, type='evaluation')
            for path in artifact_paths:
                artifact.add_file(str(path), name=Path(path).name)
            self.run.log_artifact(artifact)
        self.run.finish()
