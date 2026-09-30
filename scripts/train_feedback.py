"""Bounded adaptation pilot; frozen base, one shared feedback module."""
import argparse
import json
import random
import time

import torch
from torch.nn import functional as F
from boltzgen.model.loss.diffusion import weighted_rigid_align
from boltzgen.model.modules.utils import center_random_augmentation
from poc_model import ROOT, Feedback, LiveReadout, load_model, move, chemistry, denoise, sample


def load_cache(row, condition):
    return move(torch.load(ROOT / 'cache/features' / f'{row["id"]}-{condition}.pt', weights_only=False), 'cuda')


def loss(model, adapter, live, cache, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    kwargs = cache['conditioning']
    feats = kwargs['feats']
    design = feats['design_mask'].bool()
    target = center_random_augmentation(cache['target'], feats['atom_pad_mask'], augmentation=True)
    sigma = model.structure_module.noise_distribution(1)
    noisy = target + sigma[:, None, None] * torch.randn_like(target)
    # Include full MASK often; independently reveal chemistry at other noise levels.
    reveal = 0.0 if torch.rand(()).item() < 0.5 else torch.rand(()).item()
    chemical, hidden = chemistry(cache['labels'], design, reveal)
    pred, logits = denoise(model, adapter, live, cache, noisy, sigma, chemical)
    mask = cache['resolved'].float() * feats['atom_pad_mask'].float()
    with torch.no_grad():
        aligned = weighted_rigid_align(target.float(), pred.detach().float(), torch.ones_like(mask), mask)
    edit_atoms = torch.bmm(feats['atom_to_token'].float(), design.float()[..., None]).squeeze(-1)
    weights = mask * (0.1 + 0.9 * edit_atoms)
    mse = ((pred - aligned).square().sum(-1) * weights).sum() / (3 * weights.sum().clamp_min(1))
    xyz_loss = mse * model.structure_module.loss_weight(sigma).mean()
    ce = F.cross_entropy(logits[hidden], cache['labels'][hidden]) if hidden.any() else logits.sum() * 0
    return xyz_loss + 0.1 * ce, {'coordinate': float(xyz_loss.detach()), 'identity_ce': float(ce.detach())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(20260926)
    random.seed(20260926)
    rows = json.loads((ROOT / 'manifests/data.json').read_text())
    model = load_model()
    first = load_cache(rows['train'][0], 'flexible')
    adapter = Feedback(first['conditioning']['s_trunk'].shape[-1], model.structure_module.score_model.a_norm.normalized_shape[0]).cuda()
    print('TRAINABLE_PARAMETERS', sum(p.numel() for p in adapter.parameters()), flush=True)
    native, _ = sample(model, None, first, steps=8, seed=42)
    zero, _ = sample(model, adapter, first, steps=8, seed=42)
    maximum = float((native - zero).abs().max())
    print('ZERO_RESIDUAL_MAX_DIFFERENCE', maximum, flush=True)
    assert maximum < 1e-5, 'Zero residual must reproduce native sampler under same noise'
    live = LiveReadout(model)
    value, metrics = loss(model, adapter, live, first, seed=42)
    value.backward()
    assert adapter.residual[-1].weight.grad is not None
    assert adapter.residual[-1].weight.grad.abs().sum() > 0
    assert adapter.readout[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    assert torch.isfinite(value)
    print('GRADIENT_AUDIT_PASS', metrics, flush=True)
    adapter.zero_grad(set_to_none=True)
    out = ROOT / 'outputs/training'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps({'zero_residual_max_difference': maximum, 'gradient_pass': True,
                                             'trainable_parameters': sum(p.numel() for p in adapter.parameters())}, indent=2))
    if args.audit_only:
        return
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    train_pairs = [(r, c) for r in rows['train'] for c in ('fixed', 'flexible')]
    # Fixed validation subset, never test, for development checkpoint selection.
    val_pairs = [(r, c) for r in rows['validation'][:10] for c in ('fixed', 'flexible')]
    best = float('inf')
    started = time.time()
    order = []
    with (out / 'metrics.jsonl').open('w') as log:
        for step in range(1, args.steps + 1):
            if not order:
                order = random.sample(train_pairs, len(train_pairs))
            row, condition = order.pop()
            cache = load_cache(row, condition)
            optimizer.zero_grad(set_to_none=True)
            value, metrics = loss(model, adapter, live, cache)
            if not torch.isfinite(value):
                raise RuntimeError(f'Nonfinite training loss at {step}')
            value.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            metrics.update(step=step, condition=condition, elapsed=time.time()-started,
                           peak_vram_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(metrics)+'\n')
            log.flush()
            if step % 10 == 0:
                print('TRAIN', json.dumps(metrics), flush=True)
            if step % 100 == 0 or step == args.steps:
                scores = []
                with torch.no_grad():
                    for i, (vr, vc) in enumerate(val_pairs):
                        v, _ = loss(model, adapter, live, load_cache(vr, vc), seed=10000+i)
                        scores.append(float(v))
                score = sum(scores)/len(scores)
                record = {'adapter': adapter.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                          'token_dim': adapter.token_dim, 'hidden_dim': adapter.hidden_dim,
                          'validation_loss': score, 'elapsed_seconds': time.time()-started}
                torch.save(record, out / 'last.pt')
                if score < best:
                    best = score
                    torch.save(record, out / 'best.pt')
                print('VALIDATION', step, score, 'best', best, flush=True)
                # Validation reseeding must not repeat the following training noise stream.
                torch.manual_seed(20260926+step)
                torch.cuda.manual_seed_all(20260926+step)
    live.close()
    (out / 'COMPLETE').write_text(str(time.time()-started))


if __name__ == '__main__':
    main()
