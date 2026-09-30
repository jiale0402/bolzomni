"""One user-authorized bounded diagnostic; never optimizer.step or save weights.

Read DIAGNOSTIC_PLAN.md. Import/execution requires GPU compute and must not be
performed while the user's compute prohibition remains active.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorized-diagnostic', action='store_true')
    args = parser.parse_args()
    if not args.authorized_diagnostic:
        raise SystemExit('No model execution: separate user authorization is required.')
    assert os.environ.get('SLURM_JOB_ID'), 'Run this GPU diagnostic through Slurm only'

    import numpy as np
    import torch
    import wandb
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.task.predict.data_from_yaml import Dataset, PredictionDataset, collate
    from poc_model import ROOT as PROJECT, ASSETS, load_model, move, labels
    from model import Observations, capture_context
    from native_flow import FullOmni, losses, native_observations, refresh_teacher_bridge, sample
    from prompt import compile_prompt
    from generate_panel import decode
    from tracking import Tracker, file_hash

    root = Path(__file__).resolve().parent
    old = root.parent / 'official_poc'
    run = root / 'runs/native-1729'
    out = root / 'results/causal-diagnostic'
    # No accidental re-execution/extension of the diagnostic budget.
    if out.exists():
        raise SystemExit('Diagnostic directory already exists; inspect it, do not silently rerun.')
    out.mkdir()
    started = time.monotonic()
    deadline = started + 12 * 60
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    random.seed(1729); np.random.seed(1729); torch.manual_seed(1729)
    cfg = json.loads((run / 'config.json').read_text())
    data = json.loads((old / 'data_manifest.json').read_text())
    manifest = json.loads((root / 'benchmark_manifest.json').read_text())
    expected = 'e7d991ab7ecf0e88c78f2476ff54eeaa03c871af5ffff7a88c2ac0bae819c725'
    assert file_hash(run / 'final.pt') == expected
    for rel in ('experiments/native_condition_poc/train.py',
                'experiments/native_condition_poc/native_flow.py',
                'experiments/omni_joint/model.py', 'experiments/omni_joint/full_model.py'):
        assert file_hash(PROJECT / rel) == cfg['sources'][rel]
    tracker = Tracker(out / 'tracking', dict(checkpoint_sha256=expected,
        source_sha256=file_hash(__file__), plan_sha256=file_hash(root / 'DIAGNOSTIC_PLAN.md'),
        training=False, optimizer_steps_performed=0, new_weights_saved=False,
        hard_gpu_minutes=15, gpu=torch.cuda.get_device_name(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], train_probes=[r['id'] for r in data['train'][:4]],
        design_probes=['5YUI_0', '6E6R-short_0'],
        interpretation='Local directional evidence and paired diagnostic trajectories, not a new benchmark.'),
        'native-400-cause-diagnostic', job_type='diagnostic')
    journal = (out / 'records.jsonl').open('w', buffering=1)
    progress = []
    def record(value):
        value = dict(value, elapsed_seconds=time.monotonic() - started)
        journal.write(json.dumps(value) + '\n')
        progress.append(value)
        (out / 'progress.json').write_text(json.dumps(progress, indent=2))
        print('DIAGNOSTIC', json.dumps(value), flush=True)
    def remaining():
        return time.monotonic() < deadline

    success = False
    try:
        teacher, base = load_model(), load_model()
        def cached(row, mode):
            path = old / 'cache' / f'{row["id"]}-{mode}.pt'
            assert path.exists(), 'No automatic rebuilding of caches in this diagnostic'
            payload = move(torch.load(path, map_location='cpu', weights_only=False), 'cuda')
            refresh_teacher_bridge(teacher, payload['cache'])
            return native_observations(payload)
        first = cached(data['train'][0], 'design')
        model = FullOmni(base, teacher, first['cache']).cuda()
        ckpt = torch.load(run / 'final.pt', map_location='cpu', weights_only=False)
        assert ckpt['step'] == 400
        assert set(ckpt['adapter']) == {n for n,p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(ckpt['adapter'], strict=False)
        del ckpt, first
        model.train()
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        names, params = zip(*named)
        opt = torch.load(run / 'resume.pt', map_location='cpu', weights_only=False)
        assert opt['step'] == 400
        optstate = opt['optimizer']; del opt
        groups = {prefix: [i for i, n in enumerate(names) if n.startswith(prefix)]
                  for prefix in ('base.structure_module.', 'base.diffusion_conditioning.', 'adapter.')}
        groups['all'] = list(range(len(params)))
        parameter_states = {}
        for group, prefix in zip(optstate['param_groups'], ('base.', 'adapter.')):
            chosen = [i for i, n in enumerate(names) if n.startswith(prefix)]
            assert len(chosen) == len(group['params'])
            for i, pid in zip(chosen, group['params']):
                state = optstate['state'][pid]
                assert int(state['step']) == 400
                parameter_states[i] = (group, state)
        del optstate
        terms = ['folding_geometry', 'folding_teacher', 'design_geometry', 'design_weighted_ce', 'design_teacher']
        for probe_index, row in enumerate(data['train'][:4]):
            if not remaining(): break
            gradients = []; scalars = {}; input_hashes = {}
            for mode in ('folding', 'design'):
                payload = cached(row, mode)
                input_hashes[mode] = file_hash(old / 'cache' / f'{row["id"]}-{mode}.pt')
                parts, _ = losses(model, payload, seed=31000 + probe_index,
                    sigma_override=8., chemical_t_override=.5)
                values = [('geometry', parts['coordinate'] + parts['smooth_lddt'])]
                if mode == 'design': values.append(('weighted_ce', cfg['identity_weight'] * parts['identity']))
                values.append(('teacher', parts['replay']))
                for j, (label, value) in enumerate(values):
                    scalars[mode + '_' + label] = float(value.detach())
                    gradients.append(torch.autograd.grad(value, params,
                        retain_graph=j < len(values) - 1, allow_unused=True))
                del values, parts, payload
                model.live = None
            gram_by_group = {}
            for group_name, indices in groups.items():
                gram = torch.zeros((5, 5), device='cuda', dtype=torch.float64)
                for i in indices:
                    for a in range(5):
                        if gradients[a][i] is None: continue
                        for b in range(a, 5):
                            if gradients[b][i] is not None:
                                gram[a, b] += (gradients[a][i].float() * gradients[b][i].float()).sum().double()
                gram = gram + gram.T - torch.diag(gram.diag())
                g = gram.cpu().tolist()
                cosine = [[g[a][b] / math.sqrt(g[a][a] * g[b][b])
                    if g[a][a] > 0 and g[b][b] > 0 else None for b in range(5)] for a in range(5)]
                gram_by_group[group_name] = dict(gram=g, cosine=cosine,
                    joint_descent_directional_derivative=[-.5 * sum(r) for r in g])
            # Hypothetical next Adam direction for this fixed paired probe, using
            # saved moments. It is NOT the actual historical minibatch-400 update.
            slopes = torch.zeros(5, device='cuda', dtype=torch.float64)
            delta_norm_sq = {key: torch.zeros((), device='cuda', dtype=torch.float64)
                             for key in groups if key != 'all'}
            for i, p in enumerate(params):
                available = [g[i] for g in gradients if g[i] is not None]
                if not available: continue
                joint = available[0].float().mul(.5)
                for value in available[1:]: joint.add_(value.float(), alpha=.5)
                group, state = parameter_states[i]
                beta1, beta2 = group['betas']
                m = state['exp_avg'].to(p.device).mul(beta1).add_(joint, alpha=1-beta1)
                v = state['exp_avg_sq'].to(p.device).mul(beta2).addcmul_(joint, joint, value=1-beta2)
                step = int(state['step']) + 1
                delta = -group['lr'] * (m / (1-beta1**step)) / ((v / (1-beta2**step)).sqrt() + group['eps'])
                delta.add_(p.detach(), alpha=-group['lr'] * group['weight_decay'])
                for a in range(5):
                    if gradients[a][i] is not None:
                        slopes[a] += (gradients[a][i].float() * delta).sum().double()
                key = next(key for key in delta_norm_sq if names[i].startswith(key))
                delta_norm_sq[key] += delta.square().sum().double()
                del available, joint, m, v, delta
            record(dict(kind='gradient_probe', id=row['id'], seed=31000+probe_index,
                sigma=8., chemical_t=.5, precision='training BF16 mixed', input_hashes=input_hashes,
                terms=terms, losses=scalars, groups=gram_by_group,
                hypothetical_adam_directional_derivatives=slopes.cpu().tolist(),
                hypothetical_adam_delta_norm={k:float(v.sqrt()) for k,v in delta_norm_sq.items()},
                actual_parameter_update=False))
            del gradients
            tracker.log({f'diagnostic/probe/{k}':v for k,v in scalars.items()}, probe_index)
        del parameter_states
        torch.cuda.empty_cache()
        model.eval()
        canonicals = load_canonicals(str(ASSETS / 'mols.zip'))
        for case_index, case_id in enumerate(('5YUI_0', '6E6R-short_0')):
            if not remaining(): break
            row = next(r for r in manifest['design'] if r['id'] == case_id)
            seed = row['seed']
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            dataset = PredictionDataset(Dataset(row['yaml'], Tokenizer(atomize_modified_residues=False),
                Featurizer()), canonicals, str(ASSETS/'mols.zip'), atom14=True, design=True, disulfide_on=False)
            original_rng = np.random.default_rng
            np.random.default_rng = lambda value=None: original_rng(seed if value is None else value)
            try: feats = move(collate([dataset[0]]), 'cuda')
            finally: np.random.default_rng = original_rng
            prompt = compile_prompt(teacher, feats, canonical_frame=True)
            cache = capture_context(teacher, prompt)
            known = ~prompt['design_mask'].bool()
            identity = labels(feats); identity[~known] = 20
            obs = Observations(identity, known, torch.zeros_like(prompt['coords'].reshape(1,-1,3)),
                torch.zeros_like(prompt['atom_pad_mask'], dtype=torch.bool))
            bbmap = prompt['token_to_bb4_atoms'].float().reshape(1, -1, prompt['atom_pad_mask'].shape[1])
            editable = (~known)[0].cpu().tolist()
            def geometry_record(x, actual_state=None):
                geometric, _, _ = decode(x, prompt)
                backbone = torch.bmm(bbmap, x)[0].reshape(-1,4,3)
                ca = backbone[:,1]
                gaps = torch.linalg.vector_norm(ca[1:] - ca[:-1], dim=-1)
                true_ca = torch.bmm(prompt['token_to_rep_atom'].float(), prompt['coords'].reshape(1,-1,3))[0]
                mask = prompt['token_distance_mask'][0].bool().clone()
                mask.fill_diagonal_(False)
                from boltzgen.data import const
                letters = [const.prot_token_to_letter[a] for a in const.canonical_tokens]
                value = dict(finite=bool(torch.isfinite(x).all()), max_ca_gap=float(gaps.max()),
                    min_ca_gap=float(gaps.min()), geometric_sequence=geometric,
                    motif_ca_distance_rmse=float((torch.cdist(ca,ca)[mask]-torch.cdist(true_ca,true_ca)[mask]).square().mean().sqrt()))
                if actual_state is not None:
                    seq = ''.join(letters[i] if i < 20 else '?' for i in actual_state[0].tolist())
                    selected = [i for i,v in enumerate(editable) if v and seq[i] != '?']
                    value.update(sequence=seq, revealed=len(selected),
                        revealed_identity_geometry_agreement=sum(seq[i]==geometric[i] for i in selected)/len(selected) if selected else None)
                return value
            for condition in ('joint_feedback', 'unknowns_remain_masked'):
                if not remaining(): break
                snapshots = []; calls = [0]
                original_denoise = model.denoise
                def inspected_denoise(context, noisy, sigma, state, observations, chemical_t, native=False):
                    used = state if condition == 'joint_feedback' else observations.identity
                    pred, logits = original_denoise(context, noisy, sigma, used, observations, chemical_t, native=native)
                    if calls[0] in (0, 99, 249, 399, 499):
                        snapshots.append(dict(call=calls[0], chemical_t=float(chemical_t[0]),
                            sigma=float(torch.as_tensor(sigma).flatten()[0]), **geometry_record(pred, state)))
                    calls[0] += 1
                    return pred, logits
                model.denoise = inspected_denoise
                try:
                    with torch.no_grad(): result = sample(model, cache, obs, steps=500, seed=seed)
                finally: model.denoise = original_denoise
                x = result['coordinates']
                final = geometry_record(x, result['identity'])
                np.savez_compressed(out / f'{case_id}-{condition}.npz',
                    coordinates=x.cpu().numpy(), identity=result['identity'].cpu().numpy())
                record(dict(kind='feedback_trajectory', id=case_id, condition=condition, seed=seed,
                    snapshots=snapshots, final=final, sampling_steps=500,
                    checkpoint_sha256=expected, refolding_performed=False))
                tracker.log({f'diagnostic/{condition}/max_ca_gap':final['max_ca_gap']}, 4+case_index)
        record(dict(kind='completion', gradient_probes=sum(r['kind']=='gradient_probe' for r in progress),
            trajectories=sum(r['kind']=='feedback_trajectory' for r in progress),
            gpu_hours=(time.monotonic()-started)/3600, deadline_reached=not remaining(),
            optimizer_steps_performed=0, new_weights_saved=False))
        tracker.log({'compute/GPU_hours':(time.monotonic()-started)/3600,
            'diagnostic/gradient_probes':sum(r['kind']=='gradient_probe' for r in progress),
            'diagnostic/trajectories':sum(r['kind']=='feedback_trajectory' for r in progress)}, 6)
        artifact = wandb.Artifact('native-400-cause-diagnostic', type='diagnostics')
        for path in (out/'records.jsonl', root/'DIAGNOSTIC_PLAN.md', Path(__file__)):
            artifact.add_file(str(path))
        tracker.run.log_artifact(artifact)
        success = True
    finally:
        journal.close()
        tracker.finish(success)


if __name__ == '__main__':
    main()
