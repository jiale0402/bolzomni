"""Strict checkpoint schema validation before any GPU work."""
import json
from pathlib import Path
import torch
from fusion import Fusion
from runtime import CONFIG, ROOT, atomic_json, sha

if sha(CONFIG['checkpoint']) == CONFIG['base_checkpoint_sha256']:
    report = {'checkpoint_sha256': sha(CONFIG['checkpoint']), 'variant': 'base',
              'projection_width': 0, 'strict_state_dict_loaded': False,
              'identity': 'unchanged official AnewOmni checkpoint; no adapter'}
    atomic_json(ROOT / 'CHECKPOINT_CHECK.json', report)
    print(json.dumps(report))
    raise SystemExit(0)
checkpoint = torch.load(CONFIG['checkpoint'], map_location='cpu', weights_only=False)
if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('fusion'), dict):
    raise ValueError('Expected an Omni adapter checkpoint with a fusion state_dict. A raw Anew/BoltzGen checkpoint is not this adapter format.')
state = checkpoint['fusion']
dimensions = (state['single_mlp.0.weight'].numel(), state['pair_mlp.0.weight'].numel())
variant = {(384, 128): 'full', (512, 1024): 'control'}.get(dimensions)
if variant is None:
    raise ValueError('Unsupported adapter input dimensions: ' + str(dimensions))
width = state['single_mlp.1.weight'].shape[0]
model = Fusion(variant, width)
model.load_state_dict(state, strict=True)
for name, value in model.state_dict().items():
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite checkpoint tensor: ' + name)
report = {'checkpoint_sha256': sha(CONFIG['checkpoint']), 'variant': variant,
          'projection_width': width, 'strict_state_dict_loaded': True,
          'parameters': sum(p.numel() for p in model.parameters()),
          'epoch': checkpoint.get('epoch'), 'step': checkpoint.get('step')}
atomic_json(ROOT / 'CHECKPOINT_CHECK.json', report)
print(json.dumps(report))
