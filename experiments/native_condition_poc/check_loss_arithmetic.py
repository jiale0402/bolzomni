"""Audit saved scalar arithmetic and make six consistent fixed-probe plots.

No model, checkpoint, training, inference, or numerical model diagnostic is run.
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN, OUT = ROOT/'runs/native-1729', ROOT/'results'
source = RUN/'metrics.jsonl'
rows = [json.loads(line) for line in source.read_text().splitlines()]
cfg = json.loads((RUN/'config.json').read_text())
weight = cfg['identity_weight']
training = [r for r in rows if 'train/loss' in r]
probes = [r for r in rows if 'validation/validation_probe/design/identity' in r]
assert [r['optimizer_step'] for r in training] == list(range(1, 401))
assert [r['optimizer_step'] for r in probes] == [0, 100, 200, 300, 400]
errors = {'training_total': [], 'training_components': [], 'weighted_identity': [],
          'rolling_mean_total': [], 'fixed_probe_components': [], 'prior_plotted_total': [],
          'prior_plotted_components': [], 'prior_plotted_structural_components': []}
for row in training:
    errors['training_total'].append(abs(row['train/loss'] -
        (row['train/folding/loss'] + row['train/design/loss'])/2))
    for task in ('folding', 'design'):
        p = f'train/{task}/'
        errors['training_components'].append(abs(row[p+'loss'] -
            (row[p+'coordinate'] + row[p+'smooth_lddt'] + row[p+'weighted_identity'] + row[p+'replay'])))
        errors['weighted_identity'].append(abs(row[p+'weighted_identity'] - weight*row[p+'identity']))
for i in range(19, 400):
    window = training[i-19:i+1]
    errors['rolling_mean_total'].append(abs(fmean(r['train/loss'] for r in window) -
        (fmean(r['train/folding/loss'] for r in window) + fmean(r['train/design/loss'] for r in window))/2))

prior = json.loads((OUT/'train_validation_objective.json').read_text())
structural = json.loads((OUT/'train_validation_native_baselines.json').read_text())
assert prior['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
table = []
for i, row in enumerate(probes):
    step = row['optimizer_step']
    per_sample = json.loads((RUN/f'validation-{step}.json').read_text())
    for split in ('train', 'validation'):
        values = {}
        for task in ('folding', 'design'):
            chosen = [r for r in per_sample if r['split'] == split and r['mode'] == task]
            assert len(chosen) == (4 if split == 'train' else 16)
            p = f'validation/{split}_probe/{task}/'
            for key in ('coordinate', 'smooth_lddt', 'identity', 'replay'):
                errors['fixed_probe_components'].append(abs(row[p+key] - fmean(r[key] for r in chosen)))
            values[task] = row[p+'coordinate'] + row[p+'smooth_lddt'] + weight*row[p+'identity'] + row[p+'replay']
            errors['prior_plotted_components'].append(abs(values[task] -
                prior['curves'][task]['fixed_train' if split == 'train' else 'validation'][i]))
            errors['prior_plotted_structural_components'].append(abs(
                row[p+'coordinate'] + row[p+'smooth_lddt'] - structural['curves'][task]['geometry'][split]['omni'][i]))
        total = (values['folding'] + values['design'])/2
        errors['prior_plotted_total'].append(abs(total -
            prior['curves']['total']['fixed_train' if split == 'train' else 'validation'][i]))
        table.append(dict(split=split, update=step, folding=values['folding'], design=values['design'],
                          total=total, half_folding=values['folding']/2, half_design=values['design']/2))

max_errors = {k: max(v) for k, v in errors.items()}
assert max_errors['training_total'] < 1e-12
assert max_errors['training_components'] < 1e-6
assert max_errors['weighted_identity'] < 1e-6
assert all(max_errors[k] < 1e-12 for k in errors if k not in ('training_components', 'weighted_identity'))
audit = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             training_updates=400, rolling_windows=381, fixed_probe_split_steps=10,
             formula='total = (folding + design) / 2', identity_weight=weight,
             max_absolute_errors=max_errors, all_checks_passed=True, fixed_probe_table=table,
             scope='Saved scalar arithmetic only; does not establish model capability or gradient alignment',
             plotting_issue='Earlier multi-panel figures juxtaposed changing-batch loss, fixed-probe full loss, and structural-only loss. These must not be mixed in an arithmetic comparison.')
(OUT/'loss_arithmetic_audit.json').write_text(json.dumps(audit, indent=2))
with (OUT/'loss_arithmetic_table.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
for split in ('train', 'validation'):
    selected = [r for r in table if r['split'] == split]
    for task in ('folding', 'design', 'total'):
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        x, y = [r['update'] for r in selected], [r[task] for r in selected]
        ax.plot(x, y, 'o-', color='#176b92' if split == 'train' else '#d17628', linewidth=2)
        ax.set_title(f'{task.capitalize()} FULL loss - fixed {split.upper()} set', fontsize=15)
        ax.set_xlabel('Optimizer update'); ax.set_ylabel('Full objective (lower is better)')
        ax.grid(alpha=.18); ax.set_xlim(-8, 410)
        ax.text(.98, .95, f'{y[0]:.6f} -> {y[-1]:.6f}\nChange: {100*(y[-1]/y[0]-1):+.2f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        n = 4 if split == 'train' else 16
        fig.text(.5, .115, f'{n} fixed chains; identical noise seeds; sigma=8, chemical t=0.5. Zoomed y-axis.',
                 ha='center', fontsize=9)
        fig.text(.5, .065, 'Same records and weights for all three plots: TOTAL = (FOLDING + DESIGN) / 2.',
                 ha='center', fontsize=9)
        fig.text(.5, .02, 'Full task loss includes coordinate + smooth-lDDT + weighted identity CE + teacher MSE.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .155, 1, 1))
        fig.savefig(OUT/f'checked_{task}_{split}_loss.png', dpi=160)
        plt.close(fig)

report = ['# Loss arithmetic audit', '', 'Saved scalar logs only. No model execution.', '',
          '`total = (folding + design) / 2`; identity CE weight = '+str(weight)+'.', '',
          '| Check | Maximum absolute residual |', '|---|---:|']
report += [f'| {k} | {v:.12g} |' for k,v in max_errors.items()]
report += ['', 'All 400 training updates and all 381 complete 20-update moving windows passed.',
           'Fixed probes were checked against every saved per-sample component at 0/100/200/300/400.', '',
           '| Split | Update | Folding full loss | Design full loss | Total |', '|---|---:|---:|---:|---:|']
report += [f'| {r["split"]} | {r["update"]} | {r["folding"]:.9f} | {r["design"]:.9f} | {r["total"]:.9f} |' for r in table]
report += ['', 'The fixed-probe full totals decrease from update 0 to 400 in both splits.',
           'The earlier multi-panel figures mixed three measurement protocols visually: changing-batch training, fixed-probe full objective, and structural-only objective. Their arithmetic must be compared within the same protocol.',
           'No arithmetic discrepancy beyond floating-point rounding was found. Gradient conflict cannot explain a violation of the scalar weighted-sum identity.', '',
           'The checked_*_loss.png figures each use only one fixed split and the complete weighted objective. Native structural references remain in train_validation_native_baselines.png; no native CE or native full joint scalar is invented.']
(ROOT/'LOSS_ARITHMETIC_AUDIT.md').write_text('\n'.join(report)+'\n')
print(json.dumps({'all_checks_passed':True, 'max_absolute_errors':max_errors,
                  'endpoints':[r for r in table if r['update'] in (0,400)]}))
