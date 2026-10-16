"""Fresh fusion training through the unmodified official Anew training forward."""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
import numpy as np
import torch
from common import ROOT, CODE, CONFIG, SOURCE, manifest, seed, sha, atomic_json, save_tensor, Tracker
from learning import build_model, native_batch, features, load_input, microbatches


def state_hash(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def fixed_rng(value):
    before = rng_state()
    try:
        seed(value)
        yield
    finally:
        restore_rng(before)


def count_generated(rows):
    return sum(int(load_input(r)['generate_mask'].sum()) for r in rows)


def native_loss(model, epsilon, rows):
    batch = native_batch(rows)
    f = features(rows)
    assert [len(x['mask']) for x in f] == batch['lengths'].tolist()
    assert torch.equal(torch.cat([x['generate_mask'] for x in f]), batch['generate_mask'])
    epsilon.set_features(f)
    values = model(**batch)
    # Upstream's OOM decorator must never silently turn an OOM into a training step.
    assert isinstance(values, dict) and set(values) >= {'H', 'X', 'total'}, type(values)
    assert all(v.ndim == 0 and bool(torch.isfinite(v)) for v in values.values())
    return values, int(batch['generate_mask'].sum())


@torch.no_grad()
def evaluate(model, epsilon, rows, fixed_seed):
    was_training = model.training
    model.eval()
    sums = dict(H=0., X=0., total=0.)
    count = 0
    try:
        for i, micro in enumerate(microbatches(rows)):
            with fixed_rng(fixed_seed + i):
                values, n = native_loss(model, epsilon, micro)
            count += n
            for k in sums: sums[k] += float(values[k]) * n
    finally:
        model.train(was_training); model.autoencoder.eval()
    return {k: v / count for k, v in sums.items()} | {'generated_residues': count, 'records': len(rows)}


def module_group(name):
    return '.'.join(name.split('.')[:2]) if name.startswith('attention.') else name.split('.')[0]


def loss_formula_check(model, epsilon, rows):
    """Independently recover native targets/predictions from the actual forward."""
    captured = {}
    originals = {}
    for key in ['h', 'x']:
        trans = getattr(model.diffusion, 'trans_' + key)
        original = trans.add_noise; originals[key] = original
        def record(*a, _key=key, _fn=original, **kw):
            result = _fn(*a, **kw)
            captured[_key] = result[1]
            return result
        trans.add_noise = record
    hook = epsilon.register_forward_hook(lambda m, inputs, output: captured.update(pred=output, mask=inputs[5]))
    try:
        with torch.no_grad(), fixed_rng(42101):
            values, _ = native_loss(model, epsilon, rows)
        mask = captured['mask']
        manual = {k.upper(): (prediction[mask] - captured[k][mask]).square().sum() / mask.sum()
                  for k, prediction in zip(['h', 'x'], captured['pred'])}
        manual['total'] = manual['H'] * model.h_loss_weight + manual['X']
        for k in manual: torch.testing.assert_close(manual[k], values[k], rtol=1e-6, atol=1e-7)
        return {'native': {k: float(v) for k,v in values.items()}, 'manual': {k: float(v) for k,v in manual.items()}, 'passed': True}
    finally:
        hook.remove()
        for key, fn in originals.items(): getattr(model.diffusion, 'trans_' + key).add_noise = fn


def optimize_step(model, fusion, epsilon, frozen, rows, optimizer, gradient_totals):
    count = count_generated(rows)
    optimizer.zero_grad(set_to_none=True)
    sums = dict(H=0., X=0., total=0.)
    for micro in microbatches(rows):
        values, n = native_loss(model, epsilon, micro)
        (values['total'] * (n / count)).backward()
        for k in sums: sums[k] += float(values[k].detach()) * n
    assert all(p.grad is None for p in frozen), 'Frozen base unexpectedly received gradients'
    for name,p in fusion.named_parameters():
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all()), name
            group = module_group(name)
            gradient_totals[group] = gradient_totals.get(group, 0.) + float(p.grad.detach().abs().sum())
    norm = torch.nn.utils.clip_grad_norm_(fusion.parameters(), CONFIG['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    assert all(bool(torch.isfinite(p).all()) for p in fusion.parameters())
    return sums, count, float(norm)


def bypass_check(model,epsilon,rows):
    """Compare gamma=0 against native epsilon on the very same denoiser inputs.

    Two full VAE/condition construction passes are not a paired epsilon test.
    Save the actual native call and its RNG state and replay at that boundary.
    """
    captured={}
    def before(module,args):
        captured['args']=copy.deepcopy(args);captured['rng']=rng_state()
    def after(module,args,output):
        captured['output']=tuple(x.detach().clone() for x in output)
    hook1=epsilon.base.register_forward_pre_hook(before)
    hook2=epsilon.register_forward_hook(after)
    epsilon.gamma=0
    try:
        with torch.no_grad(),fixed_rng(72101):
            values,_=native_loss(model,epsilon,rows)
        hook1.remove();hook2.remove()
        with torch.no_grad(),fixed_rng(72101):
            restore_rng(captured['rng'])
            native=epsilon.base(*captured['args'])
        for a,b in zip(captured['output'],native):torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-5)
        return {'passed':True,'scope':'same captured denoiser inputs and RNG; native VAE and random conditions computed once',
                'max_absolute_error':[float((a-b).abs().max()) for a,b in zip(captured['output'],native)],
                'native_loss':{k:float(v) for k,v in values.items()}}
    finally:
        hook1.remove();hook2.remove();epsilon.gamma=1.

