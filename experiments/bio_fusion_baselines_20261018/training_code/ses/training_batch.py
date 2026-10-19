"""Shared production backward pass, also exercised without optimizer updates."""
from learning import load_input, microbatches
from native_checks import native_loss

MODALITIES = ['peptide', 'antibody', 'ligand']


def backward_batch(model, epsilon, frozen, group):
    n_total = sum(int(load_input(r)['generate_mask'].sum()) for r in group)
    assert n_total > 0
    sums = dict(H=0., X=0., total=0.)
    details, presentations = {}, {}
    for modality in MODALITIES:
        subset = [r for r in group if r['modality'] == modality]
        ms = dict(H=0., X=0., total=0.)
        mn = 0
        for micro in microbatches(subset, max_cost=70000, max_count=2):
            values, n = native_loss(model, epsilon, micro)
            (values['total'] * (n/n_total)).backward()
            mn += n
            for k in sums:
                v = float(values[k].detach())
                sums[k] += v*n/n_total
                ms[k] += v*n
        if mn:
            details.update({f'train/{modality}/{k.lower()}': v/mn for k, v in ms.items()})
        presentations[modality] = len(subset)
    assert all(p.grad is None for p in frozen)
    return sums, details, presentations
