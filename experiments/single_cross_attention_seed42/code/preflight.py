"""Real native checkpoint/data checks; diagnostic updates are always discarded."""
import copy
import time
import torch
from common import ROOT, atomic_json
from learning import features
from native_checks import fixed_rng, native_loss, state_hash, loss_formula_check, bypass_check


def run(model, fusion, epsilon, frozen, groups, tracker):
    start = time.monotonic()
    pristine = copy.deepcopy(fusion.state_dict())
    base_hash = state_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})
    records = []
    try:
        model.eval()
        for i, modality in enumerate(['peptide', 'antibody', 'ligand']):
            rows = sorted(groups['train'][modality], key=lambda r:r['nblocks'])
            row = rows[len(rows)//2]
            captured = {}
            hook = epsilon.register_forward_pre_hook(lambda module, args: captured.update(args=args))
            try:
                with torch.no_grad(), fixed_rng(3011+i):
                    initial, _ = native_loss(model, epsilon, [row])
            finally:
                hook.remove()
            with torch.no_grad():
                actual = epsilon(*captured['args'])
                native = epsilon.base(*captured['args'])
            errors = [float((x-y).abs().max()) for x,y in zip(actual,native)]
            for x,y in zip(actual,native):
                torch.testing.assert_close(x,y,rtol=1e-5,atol=5e-6)
            formula = loss_formula_check(model, epsilon, [row])
            bypass = bypass_check(model, epsilon, [row])
            optimizer = torch.optim.AdamW(fusion.parameters(), lr=3e-4)
            steps = []
            for step in range(2):
                model.train(); model.autoencoder.eval()
                model.zero_grad(set_to_none=True)
                with fixed_rng(4011+i):
                    values, generated = native_loss(model, epsilon, [row])
                    values['total'].backward()
                gradients = {}
                for name, p in fusion.named_parameters():
                    assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
                    key = name.split('.')[0]
                    gradients[key] = gradients.get(key, 0.)+float(p.grad.abs().sum())
                assert gradients['output'] > 0
                if step == 1:
                    assert all(v > 0 for v in gradients.values()), gradients
                assert all(p.grad is None for p in frozen)
                norm = torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                event = {'modality':modality,'diagnostic_step':step+1,'generated_blocks':generated,
                    'loss':{k:float(v.detach()) for k,v in values.items()},'grad_norm':float(norm),
                    'gradient_l1':gradients}
                steps.append(event)
                tracker.log({'smoke/'+modality+'/step':step+1, 'smoke/'+modality+'/total':event['loss']['total'],
                    'smoke/'+modality+'/grad_norm':float(norm)})
            model.eval()
            f = features([row]); gen = f[0]['generate_mask']
            epsilon.set_features(f)
            with torch.no_grad():
                out1 = epsilon(*captured['args'])
                epsilon.set_features([{**v,'single':torch.zeros_like(v['single']),
                    'pair':torch.zeros_like(v['pair'])} for v in f])
                out2 = epsilon(*captured['args'])
                effect = sum(float((a[gen]-b[gen]).abs().sum()) for a,b in zip(out1,out2))
            assert effect > 1e-8, (modality, 'donor has no generated-output effect')
            # Strict state reload and a live numerical replay after the two updates.
            checkpoint = ROOT/'profile'/('diagnostic_'+modality+'.pt')
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'fusion':fusion.state_dict()},checkpoint)
            fusion.load_state_dict(torch.load(checkpoint,weights_only=True)['fusion'],strict=True)
            epsilon.set_features(f)
            with torch.no_grad(): replay = epsilon(*captured['args'])
            for a,b in zip(out1,replay): torch.testing.assert_close(a,b,rtol=1e-5,atol=5e-6)
            records.append({'modality':modality,'key':row['key'],'blocks':row['nblocks'],
                'zero_init_errors':errors,'native_loss_formula':formula,'gamma_zero_bypass':bypass,
                'steps':steps,'generated_output_donor_effect_l1':effect,'checkpoint_reload':True})
            fusion.load_state_dict(pristine)
            model.zero_grad(set_to_none=True)
        # A real heterogeneous batch, followed by each modality's largest record.
        mixed = [min(groups['train'][m],key=lambda r:r['nblocks']) for m in ['peptide','antibody','ligand']]
        stress = [('mixed_three_modalities',mixed)]
        stress += [('largest_'+m,[max(groups['train'][m],key=lambda r:r['nblocks'])])
                   for m in ['peptide','antibody','ligand']]
        stress_records = []
        for label, rows in stress:
            model.train();model.autoencoder.eval();model.zero_grad(set_to_none=True)
            with fixed_rng(6001):
                values, generated = native_loss(model,epsilon,rows)
                values['total'].backward()
            assert all(p.grad is None for p in frozen)
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in fusion.parameters())
            stress_records.append({'case':label,'keys':[r['key'] for r in rows],
                'blocks':[r['nblocks'] for r in rows],'generated_blocks':generated,
                'total':float(values['total'].detach())})
            model.zero_grad(set_to_none=True)
        assert state_hash({n:p for n,p in model.named_parameters() if not p.requires_grad}) == base_hash
        result = {'passed':True,'records':records,'stress':stress_records,
            'diagnostic_optimizer_updates':6,'retained_optimizer_updates':0,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'fusion_parameters':sum(p.numel() for p in fusion.parameters()),
            'frozen_base_unchanged':True,'gpu':torch.cuda.get_device_name(),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.monotonic()-start}
        assert result['trainable_parameters'] == result['fusion_parameters'] == 4028016
        atomic_json(ROOT/'profile/NATIVE_PREFLIGHT.json',result)
        tracker.log({'preflight/passed':True,'preflight/parameters':result['trainable_parameters'],
            'preflight/seconds':result['seconds'],'preflight/peak_gib':result['peak_allocated_gib']})
        return result
    finally:
        fusion.load_state_dict(pristine)
        model.zero_grad(set_to_none=True)
        model.eval();model.autoencoder.eval();epsilon.gamma=1.
