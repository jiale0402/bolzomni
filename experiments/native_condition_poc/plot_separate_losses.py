"""Three standalone loss figures from saved scalars; no model execution."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/native-1729'
OUT = ROOT / 'results'
path = RUN / 'metrics.jsonl'
rows = [json.loads(line) for line in path.read_text().splitlines()]
training = [r for r in rows if 'train/loss' in r]
probes = [r for r in rows if 'validation/validation_probe/design/identity' in r]
objective = json.loads((OUT / 'train_validation_objective.json').read_text())
references = json.loads((OUT / 'train_validation_native_baselines.json').read_text())
assert objective['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
assert len(training) == 400 and len(probes) == 5
assert all(r['replay'] == 0 for r in json.loads((RUN / 'validation-0.json').read_text()))
weight = objective['identity_weight']
colors = {'train': '#176b92', 'validation': '#d17628'}
task_colors = {'folding': '#176b92', 'design': '#d17628', 'total': '#71519b'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

def common_values(task, split):
    tasks = ['folding', 'design'] if task == 'total' else [task]
    native = sum(references['curves'][t]['geometry'][split]['native_reference'] for t in tasks) / len(tasks)
    values = []
    for row in probes:
        values.append(sum(row[f'validation/{split}_probe/{t}/coordinate']
                          + row[f'validation/{split}_probe/{t}/smooth_lddt'] for t in tasks) / len(tasks))
    return native, values

outputs = []
for task in ('folding', 'design', 'total'):
    fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.9))
    ax = axs[0]
    field = 'train/loss' if task == 'total' else f'train/{task}/loss'
    x = [r['optimizer_step'] for r in training]
    y = [r[field] for r in training]
    smooth = [sum(y[i-19:i+1]) / 20 for i in range(19, len(y))]
    ax.plot(x, y, color=task_colors[task], alpha=.20, linewidth=.7, label='Raw training batch')
    ax.plot(x[19:], smooth, color=task_colors[task], linewidth=2, label='20-update mean')
    ax.set_title('Training batches: full objective', fontsize=11)
    ax.legend(fontsize=9)
    px = objective['curves'][task]['probe_steps']
    for split, data_key, label, marker in [('train', 'fixed_train', 'Train: 4 fixed chains', 'o'),
                                          ('validation', 'validation', 'Validation: 16 chains', 's')]:
        vals = objective['curves'][task][data_key]
        axs[1].plot(px, vals, color=colors[split], marker=marker, markersize=4,
                    label=f'{label}\n{vals[0]:.5f} -> {vals[-1]:.5f}')
        native, geometry = common_values(task, split)
        axs[2].axhline(native, color=colors[split], linestyle=':', linewidth=2,
                       label=f'BoltzGen ref. / {split}')
        axs[2].plot(px, geometry, color=colors[split], marker=marker, markersize=4,
                    label=f'Omni / {split}')
    axs[1].set_title('Fixed train / validation: full objective', fontsize=11)
    axs[1].legend(fontsize=9, loc='best')
    axs[2].set_title('Shared structure loss + native reference', fontsize=11)
    axs[2].legend(fontsize=8, loc='center right')
    for ax in axs:
        ax.set_xlabel('Optimizer update')
        ax.set_ylabel('Loss (lower is better)')
        ax.set_xlim(-5, 410)
        ax.grid(alpha=.16)
    fig.suptitle(f'{task.capitalize()} loss | shared Omni | 400 updates', fontsize=17, y=.97)
    formula = ('Folding = coordinate + smooth-lDDT + teacher MSE.' if task == 'folding' else
               f'Design = coordinate + smooth-lDDT + {weight:.6f} x identity CE + teacher MSE.' if task == 'design' else
               'Total = (folding full objective + design full objective) / 2; equal task weights.')
    fig.text(.5, .145, formula, ha='center', fontsize=10)
    fig.text(.5, .095,
             'Middle/right: fixed samples and noise seeds; sigma=8, chemical t=0.5. Right excludes identity CE and teacher MSE.',
             ha='center', fontsize=9)
    fig.text(.5, .048,
             'Native dotted lines are derived from archived step-0 zero-feedback probes with teacher MSE=0, not new model evaluations.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .20, 1, .89))
    dest = OUT / f'{task}_loss_separate.png'
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    outputs.append(str(dest))
print(json.dumps({'figures': outputs, 'source_sha256': objective['source_sha256'], 'model_execution': False}))
