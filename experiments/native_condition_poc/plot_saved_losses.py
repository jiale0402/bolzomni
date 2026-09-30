"""Plot saved scalar logs only; no model loading or execution."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/native-1729'
OUT = ROOT / 'results'
rows = [json.loads(line) for line in (RUN / 'metrics.jsonl').read_text().splitlines()]
cfg = json.loads((RUN / 'config.json').read_text())
weight = cfg['identity_weight']
training = [r for r in rows if 'train/loss' in r]
probes = [r for r in rows if 'validation/validation_probe/design/identity' in r]
assert [r['optimizer_step'] for r in training] == list(range(1, 401))
assert [r['optimizer_step'] for r in probes] == [0, 100, 200, 300, 400]

def objective(row, split, task):
    p = f'validation/{split}_probe/{task}/'
    return (row[p + 'coordinate'] + row[p + 'smooth_lddt']
            + weight * row[p + 'identity'] + row[p + 'replay'])

def fixed_values(split, task):
    if task == 'total':
        return [(objective(r, split, 'folding') + objective(r, split, 'design')) / 2
                for r in probes]
    return [objective(r, split, task) for r in probes]

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})
fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
colors = {'total': '#7050a0', 'folding': '#167291', 'design': '#cc6f21'}
saved = {'source_sha256': hashlib.sha256((RUN / 'metrics.jsonl').read_bytes()).hexdigest(),
         'identity_weight': weight, 'fixed_train_count': 4, 'validation_count': 16,
         'probe_sigma': 8, 'probe_chemical_t': 0.5, 'curves': {}}
for col, task in enumerate(('total', 'folding', 'design')):
    field = 'train/loss' if task == 'total' else f'train/{task}/loss'
    x = [r['optimizer_step'] for r in training]
    y = [r[field] for r in training]
    rolling = [sum(y[i-19:i+1]) / 20 for i in range(19, len(y))]
    ax = axs[0, col]
    ax.plot(x, y, color=colors[task], alpha=.18, linewidth=.6, label='Raw minibatch loss')
    ax.plot(x[19:], rolling, color=colors[task], linewidth=2, label='20-update mean')
    ax.set_title(f'{task.capitalize()} | training batches')
    ax.legend(fontsize=9)
    ax = axs[1, col]
    px = [r['optimizer_step'] for r in probes]
    train_y, val_y = fixed_values('train', task), fixed_values('validation', task)
    ax.plot(px, train_y, 'o-', color='#167291', label='Train: 4 fixed chains')
    ax.plot(px, val_y, 's--', color='#d16b25', label='Validation: 16 fixed chains')
    ax.set_title(f'{task.capitalize()} | fixed-sample evaluation')
    ax.legend(fontsize=9)
    ax.text(.03, .05, f'Train: {train_y[0]:.5f} -> {train_y[-1]:.5f}\n'
            f'Validation: {val_y[0]:.5f} -> {val_y[-1]:.5f}',
            transform=ax.transAxes, fontsize=9,
            bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    saved['curves'][task] = {'probe_steps': px, 'fixed_train': train_y, 'validation': val_y}
for ax in axs.flat:
    ax.set_xlabel('Optimizer update')
    ax.set_ylabel('Loss (lower is better)')
    ax.grid(alpha=.18)
    ax.set_xlim(0, 410)
fig.suptitle('Shared Omni: training and validation loss | 400 updates', fontsize=18, y=.98)
fig.text(.5, .067,
         'Top: changing training batches and random corruption. Bottom: same fixed samples, noise seeds, sigma=8 and chemical t=0.5.',
         ha='center', fontsize=10)
fig.text(.5, .035,
         f'Each task loss = coordinate + smooth-lDDT + {weight:.6f} x identity CE + teacher MSE. Total = (folding + design) / 2.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, .095, 1, .94))
fig.savefig(OUT / 'train_validation_objective.png', dpi=160)
plt.close(fig)
(OUT / 'train_validation_objective.json').write_text(json.dumps(saved, indent=2))
print(json.dumps(saved['curves']))
