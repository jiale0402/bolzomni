"""Disposable native-checkpoint diagnostic, included in each training budget."""
import copy
import time
import torch
from common import ROOT, CONFIG, atomic_json
from learning import features
from training_batch import backward_batch
from native_checks import fixed_rng, native_loss, state_hash, loss_formula_check, bypass_check


def run(model, fusion, epsilon, frozen, groups, tracker):
    tick = time.monotonic()
    pristine = copy.deepcopy(fusion.state_dict())
    base_hash = state_hash({n: p for n, p in epsilon.base.named_parameters()})
    records = []
    try:
        for i, modality in enumerate(['peptide', 'antibody', 'ligand']):
            rows = sorted(groups['train'][modality], key=lambda r: r['nblocks'])
            row = rows[len(rows)//2]
            captured = {}
            hook = epsilon.register_forward_pre_hook(lambda module, args: captured.update(args=args))
            model.eval()
            try:
                with torch.no_grad(), fixed_rng(3011+i):
                    initial, generated = native_loss(model, epsilon, [row])
            finally:
                hook.remove()
            formula = loss_formula_check(model, epsilon, [row])
            bypass = bypass_check(model, epsilon, [row])
            optimizer = torch.optim.AdamW([p for p in fusion.parameters() if p.requires_grad], lr=3e-4)
            steps = []
            for step in range(2):
                model.train(); model.autoencoder.eval(); model.zero_grad(set_to_none=True)
                with fixed_rng(4011+i):
                    values, generated = native_loss(model, epsilon, [row])
                    values['total'].backward()
                gradients = {}
                for name, p in fusion.named_parameters():
                    if not p.requires_grad:
                        assert p.grad is None
                        continue
                    assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
                    key = '.'.join(name.split('.')[:2])
                    gradients[key] = gradients.get(key, 0.) + float(p.grad.abs().sum())
                assert all(p.grad is None for p in frozen)
                assert all(value > 0 for value in gradients.values()), gradients
                norm = torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                steps.append(dict(step=step+1, loss={k: float(v.detach()) for k, v in values.items()},
                                  gradient_l1=gradients, grad_norm=float(norm)))
            model.eval()
            f = features([row])
            epsilon.set_features(f)
            args = list(captured['args'])
            # No native 2D/3D prompt here: test the physical latent-frame
            # equivariance of the transferred scalar operator and native EPT.
            uncond = list(args); uncond[-1] = None
            rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=args[1].device)
            shifted = list(uncond)
            shifted[1] = uncond[1] @ rotation.T + torch.tensor([1.2, -.4, .7], device=args[1].device)
            with torch.no_grad():
                first = epsilon(*args)
                plain = epsilon(*uncond)
                rotated = epsilon(*shifted)
                torch.testing.assert_close(rotated[0], plain[0], rtol=5e-4, atol=5e-4)
                torch.testing.assert_close(rotated[1], plain[1] @ rotation.T, rtol=5e-4, atol=5e-4)
                # Explicit native classifier-free guidance path, independent of
                # whether this training example happened to sample CFG.
                if args[-1] is not None:
                    cfg_args = list(args); cfg_args[-1] = copy.deepcopy(args[-1]); cfg_args[-1].w = 1.
                    guided = epsilon(*cfg_args)
                    assert all(torch.isfinite(v).all() for v in guided)
                epsilon.set_features([{**v, 'single': torch.zeros_like(v['single'])} for v in f])
                second = epsilon(*args)
                effect = sum(float((a-b).abs().sum()) for a, b in zip(first, second))
                assert effect > 1e-8, 'External donor has no generated-output effect'
                empty = [{**v, 'mask': torch.zeros_like(v['mask']), 'single': v['single'][:0],
                          'pair': v['pair'][:0, :0]} for v in f]
                epsilon.set_features(empty)
                no_donor = epsilon(*args)
                native = epsilon.base(*args)
                native_repeat = epsilon.base(*args)
                repeat_max_abs = [float((a-b).abs().max()) for a,b in zip(native, native_repeat)]
                empty_max_abs = [float((a-b).abs().max()) for a,b in zip(no_donor, native)]
                for a, b in zip(native, native_repeat):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=5e-6)
                for a, b in zip(no_donor, native):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=5e-6)
            path = ROOT / 'profile' / ('diagnostic_'+modality+'.pt')
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'fusion': fusion.state_dict()}, path)
            fusion.load_state_dict(torch.load(path, weights_only=True)['fusion'], strict=True)
            epsilon.set_features(f)
            with torch.no_grad():
                replay = epsilon(*args)
            for a, b in zip(first, replay):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=5e-6)
            records.append(dict(modality=modality, key=row['key'], initial_loss={k: float(v) for k, v in initial.items()},
                steps=steps, native_formula=formula, gamma_zero_bypass=bypass,
                generated_output_donor_effect_l1=effect, empty_donor_native_equivalence=True,
                empty_donor_max_abs=empty_max_abs, native_repeat_max_abs=repeat_max_abs,
                paired_output_tolerance=dict(rtol=1e-5, atol=5e-6),
                rotation_translation_equivariance=True, explicit_cfg_checked=args[-1] is not None,
                checkpoint_replay=True))
            fusion.load_state_dict(pristine); model.zero_grad(set_to_none=True)
        stress = []
        mixed = [min(groups['train'][m], key=lambda r: r['nblocks']) for m in ['peptide', 'antibody', 'ligand']]
        cases = [('mixed_three_modalities', mixed)] + [('largest_'+m,
            [max(groups['train'][m], key=lambda r: r['nblocks'])]) for m in ['peptide', 'antibody', 'ligand']]
        for label, rows in cases:
            model.train(); model.autoencoder.eval(); model.zero_grad(set_to_none=True)
            with fixed_rng(6001):
                values, generated = native_loss(model, epsilon, rows)
                values['total'].backward()
            assert all(p.grad is None for p in frozen)
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in fusion.parameters())
            stress.append(dict(case=label, keys=[r['key'] for r in rows], total=float(values['total'].detach())))
            model.zero_grad(set_to_none=True)
        batch = [groups['train'][m][i] for i in range(11) for m in ['peptide','antibody','ligand']][:32]
        model.train(); model.autoencoder.eval(); model.zero_grad(set_to_none=True)
        with fixed_rng(6002):
            sums, details, counts = backward_batch(model,epsilon,frozen,batch)
        gradients = [p.grad for p in fusion.parameters() if p.requires_grad]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        assert all(p.grad is None for p in frozen)
        stress.append(dict(case='effective_batch32',records=32,presentations=counts,loss=sums))
        model.zero_grad(set_to_none=True)
        assert state_hash(dict(epsilon.base.named_parameters())) == base_hash
        result = dict(passed=True, method=CONFIG['method_id'], records=records, stress=stress,
            diagnostic_optimizer_updates=6, retained_optimizer_updates=0, frozen_base_unchanged=True,
            trainable_parameters=sum(p.numel() for p in fusion.parameters() if p.requires_grad),
            total_fusion_parameters=sum(p.numel() for p in fusion.parameters()),
            no_zero_init_equivalence_claim=True, seconds=time.monotonic()-tick,
            gpu=torch.cuda.get_device_name(), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        assert result['trainable_parameters'] == CONFIG['trainable_parameters']
        atomic_json(ROOT/'profile/NATIVE_PREFLIGHT.json', result)
        tracker.log({'preflight/passed': True, 'preflight/seconds': result['seconds'],
                     'preflight/parameters': result['trainable_parameters']})
        return result
    finally:
        fusion.load_state_dict(pristine); model.zero_grad(set_to_none=True)
        model.eval(); model.autoencoder.eval(); epsilon.gamma = 1.
