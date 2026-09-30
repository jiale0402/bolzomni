"""Separate train and validation figures from saved scalar logs only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
metric_path = ROOT / 'runs/native-1729/metrics.jsonl'
rows = [json.loads(line) for line in metric_path.read_text().splitlines()]
training = [r for r in rows if 'train/loss' in r]
data = json.loads((OUT / 'train_validation_objective.json').read_text())
ref = json.loads((OUT / 'train_validation_native_baselines.json').read_text())
assert data['source_sha256'] == hashlib.sha256(metric_path.read_bytes()).hexdigest()
assert len(training) == 400
colors = {'train': '#176b92', 'validation': '#d17628'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
outputs = []

def geometry(task, split):
    tasks = ('folding', 'design') if task == 'total' else (task,)
    entries = [ref['curves'][t]['geometry'][split] for t in tasks]
    baseline = sum(r['native_reference'] for r in entries) / len(entries)
    values = [sum(r['omni'][i] for r in entries) / len(entries) for i in range(5)]
    return baseline, values

def change(values):
    return 100 * (values[-1] / values[0] - 1)

for task in ('folding', 'design', 'total'):
    for split in ('train', 'validation'):
        ncols = 3 if split == 'train' else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5.1*ncols, 4.8))
        color = colors[split]
        fixed_ax, structure_ax = axes[-2:]
        x = data['curves'][task]['probe_steps']
        values = data['curves'][task]['fixed_train' if split == 'train' else 'validation']
        if split == 'train':
            field = 'train/loss' if task == 'total' else f'train/{task}/loss'
            steps = [r['optimizer_step'] for r in training]
            losses = [r[field] for r in training]
            smoothed = [sum(losses[i-19:i+1])/20 for i in range(19, len(losses))]
            axes[0].plot(steps, losses, color=color, alpha=.2, linewidth=.7, label='Raw batch loss')
            axes[0].plot(steps[19:], smoothed, color=color, linewidth=2, label='20-update mean')
            axes[0].set_title('Actual training batches', fontsize=11)
            axes[0].legend(fontsize=9)
        count = 4 if split == 'train' else 16
        fixed_ax.plot(x, values, 'o-', color=color, linewidth=2,
                      label=f'{values[0]:.6f} -> {values[-1]:.6f}\nChange: {change(values):+.2f}%')
        fixed_ax.set_title(f'Full objective | {count} fixed {split} chains', fontsize=11)
        fixed_ax.legend(fontsize=9, loc='best')
        baseline, structure = geometry(task, split)
        structure_ax.axhline(baseline, color='#59636e', linestyle=':', linewidth=2,
                             label=f'BoltzGen reference: {baseline:.6f}')
        structure_ax.plot(x, structure, 'o-', color=color, linewidth=2,
                           label=f'Omni final: {structure[-1]:.6f}\nChange from step 0: {change(structure):+.2f}%')
        structure_ax.set_title('Structural loss | coordinate + smooth-lDDT', fontsize=11)
        structure_ax.legend(fontsize=9, loc='best')
        for ax in axes:
            ax.set_xlabel('Optimizer update')
            ax.set_ylabel('Loss (lower is better)')
            ax.set_xlim(-8, 410)
            ax.grid(alpha=.16)
        fig.suptitle(f'{task.capitalize()} loss - {split.upper()}', fontsize=18, y=.98)
        formula = ('Full objective: coordinate + smooth-lDDT + teacher MSE.' if task == 'folding'
                   else f'Full objective: coordinate + smooth-lDDT + {data["identity_weight"]:.6f} x identity CE + teacher MSE.'
                   if task == 'design' else 'Total objective = (folding objective + design objective) / 2. Structural total uses the same mean.')
        fig.text(.5, .17, formula, ha='center', fontsize=9)
        fig.text(.5, .12, f'Fixed probes: same samples and noise seeds; sigma=8, chemical t=0.5. Axes are zoomed; percentage changes are shown.',
                 ha='center', fontsize=8)
        fig.text(.5, .072, 'Native structural reference: derived from saved step-0 zero-feedback outputs with teacher MSE=0.',
                 ha='center', fontsize=8)
        fig.text(.5, .028, 'Saved logs only; no model rerun. No native baseline is assigned to the added identity CE.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .22, 1, .89))
        dest = OUT / f'{task}_{split}_loss.png'
        fig.savefig(dest, dpi=160)
        plt.close(fig)
        outputs.append(str(dest))
print(json.dumps({'outputs': outputs, 'source_sha256': data['source_sha256'], 'model_execution': False}))
