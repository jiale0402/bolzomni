"""Show exact linear addition of fixed-probe loss changes; saved scalars only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
audit = json.loads((OUT/'loss_arithmetic_audit.json').read_text())
assert audit['all_checks_passed']
rows = audit['fixed_probe_table']
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
result = {}
for split in ('train', 'validation'):
    subset = [r for r in rows if r['split'] == split]
    x = [r['update'] for r in subset]
    assert x == [0,100,200,300,400]
    for r in subset:
        assert abs(r['total']-(r['folding']+r['design'])/2)<1e-12
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    result[split] = {}
    styles = [('folding','#176b92','o'),('design','#d17628','s'),('total','#71519b','D')]
    for task, color, marker in styles:
        y = [r[task] for r in subset]
        delta = [v-y[0] for v in y]
        pct = 100*(y[-1]/y[0]-1)
        axes[0].plot(x,y,color=color,marker=marker,linewidth=2,label=f'{task.capitalize()}: {y[0]:.5f} -> {y[-1]:.5f}')
        axes[1].plot(x,delta,color=color,marker=marker,linewidth=2,
                     label=f'{task.capitalize()}: {delta[-1]:+.6f} ({pct:+.2f}%)')
        result[split][task] = {'initial':y[0], 'final':y[-1], 'delta':delta[-1], 'percent_change':pct}
    axes[0].set_ylim(0,1.05)
    axes[0].set_title('Actual losses | one shared y-axis')
    axes[0].set_ylabel('Full objective')
    axes[0].legend(fontsize=9,loc='center right')
    axes[1].set_ylim(-.085,.005)
    axes[1].set_title('Change from step 0 | one shared y-axis')
    axes[1].set_ylabel('Loss(step) - loss(0); negative = decrease')
    axes[1].axhline(0,color='#a0a0a0',linewidth=.7)
    axes[1].legend(fontsize=9,loc='lower left')
    for ax in axes:
        ax.set_xlim(-8,410);ax.set_xlabel('Optimizer update');ax.grid(alpha=.17)
    count = 4 if split == 'train' else 16
    fig.suptitle(f'{split.upper()} | {count} fixed chains | same objective and noise seeds',fontsize=16,y=.98)
    fig.text(.5,.115,'At every point: TOTAL = (FOLDING + DESIGN) / 2. The same identity holds for changes from step 0.',
             ha='center',fontsize=10)
    fig.text(.5,.067,'Both TRAIN and VALIDATION figures use the same axis limits. Every saved point is shown; no smoothing or fitted curves.',
             ha='center',fontsize=8.5)
    fig.text(.5,.022,'Step 0 denotes initial Omni. Full objectives include coordinate, smooth-lDDT, weighted identity CE and teacher MSE.',
             ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.16,1,.91))
    fig.savefig(OUT/f'{split}_loss_same_scale.png',dpi=160)
    plt.close(fig)
print(json.dumps(result))
