"""Recover common structural-loss references from archived zero-feedback probes.

No checkpoints or models are loaded. Native references are explicitly derived,
not represented as a newly executed standalone native-model evaluation.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/native-1729'
OUT = ROOT / 'results'
raw_path = RUN / 'metrics.jsonl'
rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
probes = [r for r in rows if 'validation/validation_probe/design/identity' in r]
assert [r['optimizer_step'] for r in probes] == [0, 100, 200, 300, 400]
initial = json.loads((RUN / 'validation-0.json').read_text())
assert len(initial) == 40
assert all(r['replay'] == 0. for r in initial)
mechanism = json.loads((OUT / 'mechanism_audit.json').read_text())
assert all(p['initial_norm'] == 0. for p in mechanism['projections'].values())
# Source audit: both output projections and the token output bias initialize to zero.
source = (ROOT.parent / 'omni_joint/model.py').read_text()
assert 'nn.init.zeros_(self.token_out.weight); nn.init.zeros_(self.token_out.bias)' in source
assert 'nn.init.zeros_(self.pair_out.weight)' in source
cfg = json.loads((RUN / 'config.json').read_text())
for p in [ROOT.parent / 'omni_joint/model.py', ROOT / 'native_flow.py']:
    relative = p.relative_to(ROOT.parent.parent).as_posix()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == cfg['sources'][relative]

saved = {
    'reference_type': 'Derived native-equivalent structural loss, not a separate new evaluation',
    'basis': ['Feedback output projections/bias initialize to zero',
              'Archived step-0 teacher MSE equals zero for all 40 fixed probes',
              'At initialization, denoiser output is independent of added chemical state',
              'Same resolved atoms, target alignment, noise, and structural-loss formula'],
    'scope': 'Fixed-probe structural losses only; not full native generation equivalence',
    'not_applicable': ['Added sequence-head CE', 'Omni total containing added sequence-head CE'],
    'source_sha256': {str(p.relative_to(ROOT.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [raw_path, RUN / 'validation-0.json', OUT / 'mechanism_audit.json',
                                ROOT.parent / 'omni_joint/model.py', ROOT / 'native_flow.py']},
    'steps': [r['optimizer_step'] for r in probes], 'curves': {}
}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
panels = [('geometry', 'Coordinate + smooth-lDDT'), ('coordinate', 'Coordinate MSE'),
          ('smooth_lddt', 'Smooth-lDDT')]
colors = {'train': '#167291', 'validation': '#cf6e25'}
for row_idx, task in enumerate(('folding', 'design')):
    saved['curves'][task] = {}
    for col, (key, title) in enumerate(panels):
        ax = axs[row_idx, col]
        saved['curves'][task][key] = {}
        for split, label in [('train', 'Train 4'), ('validation', 'Validation 16')]:
            prefix = f'validation/{split}_probe/{task}/'
            vals = [r[prefix+'coordinate'] + r[prefix+'smooth_lddt'] if key == 'geometry'
                    else r[prefix+key] for r in probes]
            native = vals[0]
            # Distinct marker shapes separate trained trajectories from reference lines.
            ax.axhline(native, color=colors[split], linestyle=':', linewidth=2,
                       label=f'BoltzGen reference / {label}')
            ax.plot(saved['steps'], vals, 'o-' if split == 'train' else 's-',
                    color=colors[split], linewidth=1.5, markersize=4,
                    label=f'Omni / {label}')
            saved['curves'][task][key][split] = {'native_reference': native, 'omni': vals}
        ax.set_title(f'{task.capitalize()} | {title}', fontweight='bold')
        ax.set_xlabel('Optimizer update')
        ax.set_ylabel('Loss (lower is better)')
        ax.grid(alpha=.15)
        ax.set_xlim(-8, 410)
        ax.legend(fontsize=8, loc='center right', framealpha=.9)
        tr = saved['curves'][task][key]['train']
        va = saved['curves'][task][key]['validation']
        ax.text(.02, .025, f'Native -> final Omni\nTrain: {tr["native_reference"]:.6f} -> {tr["omni"][-1]:.6f}\n'
                f'Val: {va["native_reference"]:.6f} -> {va["omni"][-1]:.6f}',
                transform=ax.transAxes, fontsize=8,
                bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
fig.suptitle('Common structural losses: Omni curves and BoltzGen reference lines', fontsize=17, y=.98)
fig.text(.5, .068,
         'Dotted horizontal lines: native-equivalent reference derived from step 0 (zero feedback; saved teacher MSE = 0 on all 40 probes).',
         ha='center', fontsize=9)
fig.text(.5, .042,
         'Same fixed train/validation chains, noise seeds, sigma=8; chemical t=0.5 for Omni. Geometry excludes identity CE and teacher regularization.',
         ha='center', fontsize=9)
fig.text(.5, .016,
         'No model was rerun. The added sequence CE and full Omni joint objective have no directly comparable native scalar baseline.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .10, 1, .94))
fig.savefig(OUT / 'train_validation_native_baselines.png', dpi=160)
plt.close(fig)
(OUT / 'train_validation_native_baselines.json').write_text(json.dumps(saved, indent=2))
print(json.dumps({'output': str(OUT / 'train_validation_native_baselines.png'),
                  'reference_type': saved['reference_type'], 'zero_replay_probes': len(initial)}))
