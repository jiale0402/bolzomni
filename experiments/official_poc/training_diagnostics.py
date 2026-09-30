"""Plot recorded training diagnostics without interpreting them as benchmark gains."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/corrected-1729'


def main():
    rows = [json.loads(line) for line in (RUN / 'metrics.jsonl').read_text().splitlines()]
    updates = [r for r in rows if 'train/loss' in r]
    assert [r['optimizer_step'] for r in updates] == list(range(1, 401))
    x = np.arange(1, 401)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for ax, key, title in [(axes[0, 0], 'train/loss', 'Joint training loss'),
                           (axes[0, 1], 'train/gradient_norm_before_clip', 'Gradient norm before clipping')]:
        y = np.asarray([r[key] for r in updates])
        assert np.isfinite(y).all()
        ax.plot(x, y, color='#386cb0', alpha=.25, linewidth=.7, label='Each update')
        ax.plot(x[24:], np.convolve(y, np.ones(25)/25, mode='valid'), color='#183e70', label='25-update mean')
        ax.set_title(title)
        ax.legend(frameon=False)
    axes[0, 1].axhline(10, color='#bd5343', linewidth=.8, linestyle='--', label='Clip threshold')
    axes[0, 1].legend(frameon=False)
    probes = []
    for step in (0, 100, 200, 300, 400):
        path = RUN / f'validation-{step}.json'
        if path.exists():
            probes.append((step, json.loads(path.read_text())))
    for mode, color in [('folding', '#386cb0'), ('design', '#c45b28')]:
        y = [np.mean([r['coordinate'] for r in records if r['split'] == 'validation' and r['mode'] == mode]) for step, records in probes]
        axes[1, 0].plot([s for s, _ in probes], y, 'o-', color=color, label=mode)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Fixed validation coordinate MSE (log scale)')
    axes[1, 0].legend(frameon=False)
    for split, color in [('train', '#386cb0'), ('validation', '#c45b28')]:
        y = [np.mean([r['identity'] for r in records if r['split'] == split and r['mode'] == 'design']) for step, records in probes]
        axes[1, 1].plot([s for s, _ in probes], y, 'o-', color=color, label=split)
    axes[1, 1].set_title('Fixed design masked identity CE')
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.set_xlabel('Optimizer update')
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Official-data Omni PoC: training diagnostics, not capability scores', fontsize=12)
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    fig.savefig(output / 'training_diagnostics.png', dpi=180)
    windows = {}
    for a, b in [(1, 40), (41, 100), (101, 200), (201, 300), (301, 400)]:
        chosen = updates[a-1:b]
        windows[f'{a}-{b}'] = {k: float(np.mean([r[k] for r in chosen])) for k in
            ('train/loss', 'train/folding/loss', 'train/design/loss', 'train/gradient_norm_before_clip', 'train/gradient_clip_factor')}
    (output / 'training_diagnostics.json').write_text(json.dumps(dict(windows=windows,
        nonfinite_updates=0, clipped_updates=sum(r['train/gradient_clip_factor'] < 1 for r in updates),
        validation_steps=[s for s, _ in probes]), indent=2))
    print(json.dumps(windows, indent=2))


if __name__ == '__main__':
    main()
