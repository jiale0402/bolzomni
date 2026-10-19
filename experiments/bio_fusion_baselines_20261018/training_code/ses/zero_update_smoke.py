"""Real-checkpoint training-path checks; zero optimizer steps on CPU or GPU.

Uses the production backward_batch, original data and losses. CPU checks do
not certify CUDA, the full batch-32 memory envelope, generation or scoring.
"""
import argparse
import copy
import json
import os
import time
import traceback
import torch
from common import ROOT, CODE, CONFIG, CKPT, Tracker, sha, seed, manifest, atomic_json
import learning
import native_checks as checks
from fusion import Fusion, FusionEpsilon
from training_batch import backward_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--seed', type=int, choices=[1, 123, 42], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    seed(args.seed)
    stage = f'zero-update-{args.device}-seed{args.seed}'
    report_path = ROOT/'profile'/(stage+'.json')
    assert not report_path.exists(), 'Preserve existing checks; use a fresh diagnostic directory'
    assert CONFIG['train_backbone'] is False
    assert sha(CKPT) == CONFIG['anew_checkpoint_sha256']
    assert sha(ROOT/'manifests/manifest.json') == CONFIG['manifest_sha256']
    full = manifest()
    assert all(len(full['train'][m]) == 10000 for m in ['peptide', 'antibody', 'ligand'])
    subset = json.loads((ROOT/'smoke_records.json').read_text())
    groups = {}
    for m, paths in subset.items():
        original = {r['key']: r for r in full['train'][m]}
        groups[m] = []
        for key, paths_for_key in paths.items():
            for kind in ['input', 'feature']:
                assert sha(paths_for_key[kind+'_path']) == paths_for_key[kind+'_sha256']
            groups[m].append({**original[key], 'modality':m,
                'input_path': paths_for_key['input_path'], 'feature_path': paths_for_key['feature_path']})
    # Explicit device wiring for this diagnostic only. Production helpers and
    # the original native forward/loss retain their implementations.
    native_batch, features = learning.native_batch, learning.features
    checks.native_batch = lambda rows: native_batch(rows, device=args.device)
    checks.features = lambda rows: features(rows, device=args.device)
    tracker = Tracker(stage, dict(seed=args.seed, device=args.device, formal_training=False,
        optimizer_updates=0, scope='Real min/median/max training records; no full training or benchmark',
        source_hashes={p.name:sha(p) for p in CODE.glob('*.py')}))
    tick = time.monotonic()
    report = dict(passed=False, optimizer_updates=0, seed=args.seed, device=args.device,
                  formal_training_started=False, full_pipeline_passed=False)
    try:
        model, _ = learning.load_official()
        model.eval().requires_grad_(False).to(args.device)
        frozen = list(model.parameters())
        fusion = Fusion(**CONFIG.get('fusion_kwargs', {})).to(args.device)
        epsilon = FusionEpsilon(model.diffusion.eps_net, fusion).to(args.device)
        model.diffusion.eps_net = epsilon
        initial = checks.state_hash(fusion.state_dict())
        base_hash = checks.state_hash(dict(epsilon.base.named_parameters()))
        optimizer = torch.optim.AdamW([{'params':[p for p in fusion.parameters() if p.requires_grad],
            'lr':CONFIG['lr'], 'name':'fusion'}], weight_decay=CONFIG['weight_decay'])
        assert {id(p) for g in optimizer.param_groups for p in g['params']} == {
            id(p) for p in model.parameters() if p.requires_grad}
        records = []
        for m in groups:
            row = min(groups[m], key=lambda r:r['nblocks'])
            model.eval()
            formula = checks.loss_formula_check(model, epsilon, [row])
            bypass = checks.bypass_check(model, epsilon, [row])
            model.train(); model.autoencoder.eval(); model.zero_grad(set_to_none=True)
            with checks.fixed_rng(args.seed):
                losses, details, presentations = backward_batch(model, epsilon, frozen, [row])
            gradients = {}
            for n,p in fusion.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
                    gradients[n] = float(p.grad.abs().sum())
            assert all(v > 0 for v in gradients.values()), gradients
            norm = torch.nn.utils.clip_grad_norm_([p for p in fusion.parameters() if p.requires_grad],
                CONFIG['grad_clip'], error_if_nonfinite=True)
            assert checks.state_hash(fusion.state_dict()) == initial
            records.append(dict(modality=m, key=row['key'], blocks=row['nblocks'], losses=losses,
                details=details, presentations=presentations, native_formula=formula,
                gamma_zero_bypass=bypass, gradient_norm=float(norm), gradient_l1=gradients))
            tracker.log({'checks/modality':m, 'checks/gradient_norm':float(norm),
                         **{'checks/'+k:v for k,v in losses.items()}})
        mixed = [min(groups[m],key=lambda r:r['nblocks']) for m in groups]
        model.train(); model.autoencoder.eval(); model.zero_grad(set_to_none=True)
        with checks.fixed_rng(args.seed+100):
            losses, details, presentations = backward_batch(model, epsilon, frozen, mixed)
        assert checks.state_hash(fusion.state_dict()) == initial
        assert checks.state_hash(dict(epsilon.base.named_parameters())) == base_hash
        checkpoint = ROOT/'profile'/f'untrained_seed{args.seed}_{args.device}.pt'
        torch.save({'fusion':fusion.state_dict(), 'optimizer':optimizer.state_dict(),
                    'step':0, 'seed':args.seed, 'untrained_diagnostic':True}, checkpoint)
        saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
        fusion.load_state_dict(saved['fusion'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        assert not optimizer.state
        assert checks.state_hash(fusion.state_dict()) == initial
        model.eval(); epsilon.set_features(features([mixed[0]],device=args.device))
        captured={}
        hook=epsilon.register_forward_pre_hook(lambda module,a:captured.update(args=a))
        with torch.no_grad(),checks.fixed_rng(1123): checks.native_loss(model,epsilon,[mixed[0]])
        hook.remove()
        with torch.no_grad():before=epsilon(*captured['args'])
        fusion.load_state_dict(saved['fusion'],strict=True)
        with torch.no_grad():after=epsilon(*captured['args'])
        for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        assert all(p.grad is None for p in frozen)
        report.update(passed=True, records=records, mixed=losses, mixed_presentations=presentations,
            frozen_base_unchanged=True, fusion_weights_unchanged=True, strict_checkpoint_replay=True,
            optimizer_parameter_coverage=True, initial_adapter_hash=initial,
            checkpoint_sha256=sha(checkpoint), training_batch_sha256=sha(CODE/'training_batch.py'),
            seconds=time.monotonic()-tick, trainable_parameters=sum(p.numel() for p in fusion.parameters() if p.requires_grad),
            gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
            untested=['Full batch-32 memory stress','Generation/export/scoring integration']+
                (['CUDA execution'] if args.device=='cpu' else []))
        atomic_json(report_path,report)
        tracker.finish(report,[report_path,checkpoint])
        print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc(), seconds=time.monotonic()-tick)
        atomic_json(report_path,report)
        tracker.log({'checks/failed':True,'checks/error':str(exc)})
        tracker.run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
