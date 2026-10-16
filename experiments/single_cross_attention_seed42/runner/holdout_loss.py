"""Reference-supervised benchmark loss, independent of checkpoint selection.

Called after training selection and before the fixed generation pilot. No
optimizer, no feature extraction, and no generated candidate is used as a label.
"""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

MODALITIES = ('peptide', 'antibody', 'ligand')
METRICS = ('H', 'X', 'total')
ROOT = Path('/projects/p53127/anewomni_work/trimodal_holdout_loss_20261010')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_checkpoint(model, epsilon, identity, validation, rows, runtime):
    from data.base import BaseDataset
    out = ROOT / 'results' / identity['name']
    overlap_audit = json.loads((ROOT / 'PDB_OVERLAP_AUDIT.json').read_text())
    overlap_targets = {r['target'] for b in overlap_audit['benchmarks'] for r in b['matches']}
    assert len(overlap_targets) == 19
    config = dict(identity, split='benchmark_holdout', targets=253,
                  modality_records={'peptide': 93, 'antibody': 60, 'ligand': 100},
                  draws_per_target=4, noise_protocol='Native random time/noise/conditioning; four fixed per-target seeds shared across models',
                  seed_base=20261010, reduction='generated blocks, matching native validation objective',
                  checkpoint_selection='Already locked by validation; never selected using this loss',
                  supervision='Native model.forward on real reference structures, with original condition sampler',
                  donor='Existing sanitized generation cache; reference answer never passed to donor',
                  manifest_sha256=sha(ROOT / 'manifest.json'), source_sha256=sha(__file__),
                  overlap_audit_sha256=sha(ROOT / 'PDB_OVERLAP_AUDIT.json'),
                  validation=validation, known_training_pdb_overlap_ligand_targets=19,
                  benchmark_previously_inspected=True, training_updates=0)
    if (out / 'COMPLETE.json').exists():
        report = json.loads((out / 'COMPLETE.json').read_text())
        assert report['protocol'] == config
        return report
    assert not (out / 'STARTED.json').exists(), 'Audit interrupted evaluation before retrying'
    write(out / 'STARTED.json', config)
    # One independent W&B evaluation run for each frozen checkpoint. Actual
    # training curves remain in their original runs, without substituted probes.
    import wandb
    run_id = 'holdout-' + identity['checkpoint_sha256'][:12] + '-' + config['manifest_sha256'][:8]
    kwargs = dict(project=runtime.CONFIG['wandb_project'], entity=runtime.CONFIG['wandb_entity'],
                  id=run_id, name=identity['name'] + '-benchmark-holdout-loss',
                  group='trimodal-benchmark-holdout-20261010', job_type='benchmark-holdout-loss',
                  config=config, dir=str(out), resume='allow')
    try:
        run = wandb.init(**kwargs, settings=wandb.Settings(init_timeout=30))
    except Exception as exc:
        print('WANDB_OFFLINE_FALLBACK', type(exc).__name__, flush=True)
        wandb.finish()
        run = wandb.init(**kwargs, mode='offline')
    online = run.settings.mode == 'online'
    write(out / 'WANDB_STATUS.json', {'online': online, 'pending_sync': not online,
                                     'run_id': run_id, 'url': run.url if online else None})
    run.define_metric('checkpoint_step')
    for prefix in ['holdout/*', 'validation/*']:
        run.define_metric(prefix, step_metric='checkpoint_step')
    sums = {m: dict.fromkeys(METRICS, 0.) for m in MODALITIES}
    blocks = dict.fromkeys(MODALITIES, 0)
    counts = dict.fromkeys(MODALITIES, 0)
    clean_sums = {m: dict.fromkeys(METRICS, 0.) for m in MODALITIES}
    clean_blocks, clean_counts = dict.fromkeys(MODALITIES, 0), dict.fromkeys(MODALITIES, 0)
    started = time.monotonic()
    was_training = model.training
    model.eval()
    before = {n: p._version for n, p in model.named_parameters()}
    seen = 0
    try:
        with torch.inference_mode():
            for row in rows:
                assert sha(row['input_path']) == row['input_sha256']
                saved = torch.load(row['input_path'], map_location='cpu', weights_only=False)
                if epsilon is not None:
                    assert sha(row['feature_path']) == row['feature_sha256']
                    f = torch.load(row['feature_path'], map_location='cpu', weights_only=True)
                    f = {'single': f['single'], 'pair': f['pair'], 'mask': saved['coverage'],
                         'generate_mask': saved['data']['generate_mask']}
                    assert len(f['single']) == int(f['mask'].sum())
                    epsilon.set_features([{k: v.cuda() for k, v in f.items()}], cache=False)
                m = row['modality']
                values_per_target = []
                for draw in range(4):
                    seed = int.from_bytes(hashlib.sha256(f"holdout:20261010:{row['key']}:{draw}".encode()).digest()[:4], 'big')
                    runtime.seed(seed)
                    # Fresh copy because native collation updates bond indices.
                    data = {k: v.clone() if torch.is_tensor(v) else v for k, v in saved['data'].items()}
                    batch = BaseDataset.collate_fn(None, [data])
                    batch = {k: v.cuda() for k, v in batch.items()}
                    result = model(**batch)
                    assert isinstance(result, dict) and set(result) >= set(METRICS)
                    values = {k: float(result[k]) for k in METRICS}
                    assert all(np.isfinite(v) for v in values.values())
                    assert abs(values['total'] - (model.h_loss_weight * values['H'] + values['X'])) < 2e-5
                    values_per_target.append(values)
                    record = {'target': row['target'], 'key': row['key'], 'modality': m, 'draw': draw,
                              'seed': seed, 'generated_blocks': row['generated_blocks'], **values}
                    with (out / 'records.jsonl').open('a') as handle:
                        handle.write(json.dumps(record, allow_nan=False) + '\n')
                means = {k: sum(v[k] for v in values_per_target) / 4 for k in METRICS}
                n = row['generated_blocks']
                for k in METRICS:
                    sums[m][k] += means[k] * n
                counts[m] += 1
                blocks[m] += n
                if row['target'] not in overlap_targets:
                    for k in METRICS:
                        clean_sums[m][k] += means[k] * n
                    clean_blocks[m] += n
                    clean_counts[m] += 1
                seen += 1
                if seen % 25 == 0:
                    print('HOLDOUT', identity['name'], seen, len(rows), round(time.monotonic() - started, 1), flush=True)
        assert counts == {'peptide': 93, 'antibody': 60, 'ligand': 100}
        assert before == {n: p._version for n, p in model.named_parameters()}
        results = {m: {k: sums[m][k] / blocks[m] for k in METRICS} |
                   {'records': counts[m], 'generated_blocks': blocks[m], 'noise_evaluations': counts[m] * 4}
                   for m in MODALITIES}
        results['combined'] = {k: sum(sums[m][k] for m in MODALITIES) / sum(blocks.values()) for k in METRICS}
        clean = {m: {k: clean_sums[m][k] / clean_blocks[m] for k in METRICS} |
                 {'records': clean_counts[m], 'generated_blocks': clean_blocks[m]}
                 for m in MODALITIES}
        assert clean_counts == {'peptide': 93, 'antibody': 60, 'ligand': 81}
        log = {'checkpoint_step': identity['checkpoint_step'],
               **{f'holdout/{m}/{k.lower()}': v for m, vs in results.items() for k, v in vs.items()},
               **{f'holdout_no_pdb_overlap/{m}/{k.lower()}': v for m, vs in clean.items() for k, v in vs.items()},
               **{f'validation/{m}/{k.lower()}': v for m, vs in validation.items() for k, v in vs.items()}}
        with (out / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(log, allow_nan=False) + '\n')
        run.log(log)
        report = {'completed': True, 'protocol': config, 'results': results, 'validation': validation,
                  'results_no_detected_training_pdb_overlap': clean,
                  'seconds': time.monotonic() - started, 'gpu': torch.cuda.get_device_name(),
                  'training_updates': 0, 'parameter_versions_unchanged': True,
                  'wandb_online': online, 'pending_sync': not online,
                  'records_sha256': sha(out / 'records.jsonl')}
        write(out / 'RESULT.json', report)
        run.summary.update(log | {'completed': True, 'evaluation_seconds': report['seconds']})
        artifact = wandb.Artifact(run_id + '-evidence', type='evaluation')
        for name in ['RESULT.json', 'records.jsonl', 'metrics.jsonl', 'STARTED.json']:
            artifact.add_file(str(out / name), name=name)
        run.log_artifact(artifact)
        run.finish()
        write(out / 'COMPLETE.json', report)
        return report
    except BaseException as exc:
        import traceback
        write(out / 'FAILED.json', {'error': str(exc), 'traceback': traceback.format_exc(),
                                   'targets_completed': seen, 'seconds': time.monotonic() - started})
        run.finish(exit_code=1)
        raise
    finally:
        model.train(was_training)
        model.autoencoder.eval()


def run_all(model, current_epsilon, runtime):
    """Evaluate all existing references; restore the selected generation model."""
    prepared = json.loads((ROOT / 'PREPARED.json').read_text())
    assert prepared['completed'] and sha(ROOT / 'manifest.json') == prepared['manifest_sha256']
    assert json.loads((ROOT / 'CPU_PREFLIGHT.json').read_text())['passed']
    rows = json.loads((ROOT / 'manifest.json').read_text())
    assert len(rows) == 253
    base = current_epsilon.base
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    study = Path(runtime.CONFIG['output']).parent
    training = json.loads((study / 'pair_bias/training/COMPLETE.json').read_text())
    old_root = Path('/projects/p53127/anewomni_work/trimodal_interface_20261010')
    old_training = json.loads((old_root / 'hidden/training/COMPLETE.json').read_text())
    spec = import_file('holdout_hidden_fusion', old_root / 'hidden/code/fusion.py')
    old_checkpoint = Path('/projects/p53127/anewomni_work/trimodal_refit_20261009/refine_selected_cosine/training/best_joint.pt')
    assert sha(old_checkpoint) == '3bdd6da6a829843d697a24f335dee339d2100497c555168298a5ce96abd5e12b'
    settings = [
        ('pair_bias_1h', current_epsilon, Path(training['checkpoint']), training['optimizer_step'], training['validation']),
        ('anew_original', None, Path(runtime.CONFIG['runtime']['base_checkpoint']), 0,
         json.loads((study / 'pair_bias/training/initial_validation.json').read_text())),
    ]
    # Historical checkpoints are read only. No old interface is retrained.
    for name, interface, checkpoint, val, step in [
        ('old_full', 'conditioning', old_checkpoint, None, None),
        ('hidden_1h', 'hidden', Path(old_training['checkpoint']), old_training['validation'], old_training['optimizer_step']),
    ]:
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        fusion = spec.Fusion().cuda()
        fusion.load_state_dict(ckpt['fusion'], strict=True)
        wrapper = spec.FusionEpsilon(base, fusion, interface=interface).cuda().eval().requires_grad_(False)
        settings.append((name, wrapper, checkpoint, step if step is not None else ckpt.get('step', 0),
                         val if val is not None else ckpt['records'][-1]['validation']))
    reports = {}
    try:
        for name, wrapper, checkpoint, step, val in settings:
            model.diffusion.eps_net = base if wrapper is None else wrapper
            identity = {'name': name, 'checkpoint_sha256': sha(checkpoint), 'checkpoint_path': str(checkpoint),
                        'checkpoint_step': step}
            reports[name] = evaluate_checkpoint(model, wrapper, identity, val, rows, runtime)
        write(ROOT / 'COMPLETE.json', {'completed': True, 'models': list(reports),
                                      'targets_per_model': 253, 'training_updates': 0,
                                      'results': {k: v['results'] for k, v in reports.items()}})
        render_report(reports)
    finally:
        model.diffusion.eps_net = current_epsilon
        model.eval()
        random.setstate(state[0]); np.random.set_state(state[1])
        torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])
    return reports


def render_report(reports):
    import html
    order = ['anew_original', 'old_full', 'hidden_1h', 'pair_bias_1h']
    labels = {'peptide': '肽', 'antibody': '抗体', 'ligand': '小分子'}
    raw = {}
    for name in order:
        grouped = defaultdict(list)
        for line in (ROOT / 'results' / name / 'records.jsonl').read_text().splitlines():
            r = json.loads(line)
            grouped[(r['modality'], r['key'])].append(r)
        assert len(grouped) == 253 and all(len(rs) == 4 for rs in grouped.values())
        raw[name] = {key: {k: sum(r[k] for r in rs) / 4 for k in METRICS} |
                     {'blocks': rs[0]['generated_blocks']} for key, rs in grouped.items()}
    contrasts = []
    rng = np.random.default_rng(20261010)
    for m in MODALITIES:
        keys = sorted(k for k in raw[order[0]] if k[0] == m)
        weights = np.array([raw[order[0]][k]['blocks'] for k in keys])
        draws = rng.integers(len(keys), size=(20000, len(keys)))
        for metric in METRICS:
            for ref in ['anew_original', 'old_full']:
                for name in ['old_full', 'hidden_1h', 'pair_bias_1h']:
                    if ref == name:
                        continue
                    assert all(raw[name][k]['blocks'] == raw[ref][k]['blocks'] for k in keys)
                    delta = np.array([raw[name][k][metric] - raw[ref][k][metric] for k in keys]) * weights
                    boot = delta[draws].sum(axis=1) / weights[draws].sum(axis=1)
                    contrasts.append({'modality': m, 'metric': metric, 'model': name, 'reference': ref,
                                      'delta': float(delta.sum() / weights.sum()),
                                      'paired_target_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()})
    write(ROOT / 'COMPARISON.json', {'completed': True, 'reports': reports, 'contrasts': contrasts,
                                   'intervals': 'Unadjusted, paired target bootstrap, conditional on four fixed noise draws and frozen checkpoints'})
    parts = ['<!doctype html><meta charset="utf-8"><title>Validation 与 benchmark holdout loss</title>',
             '<style>body{font:16px system-ui;max-width:1150px;margin:35px auto;color:#17212b}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{border:1px solid #d3d8de;padding:10px;text-align:right}th:first-child,td:first-child{text-align:left}small{color:#52606d}</style>',
             '<h1>Validation 与 benchmark holdout loss</h1>',
             '<p>所有 loss 越低越好。Validation 为既有 614 条选模集；benchmark holdout 为 253 个真实参考结构，4 组固定噪声/目标，按生成块数加权。Holdout 不用于选 checkpoint。</p>',
             '<p>Benchmark 已被查看；其中 19 个小分子目标检测到训练 PDB 重叠。以下 loss 不是生成质量指标，也不是未经接触的独立测试集。</p>']
    for split, field in [('Validation（614 条）', 'validation'), ('Benchmark holdout（253 个目标）', 'results')]:
        parts.append('<h2>' + split + '</h2><table><tr><th>模态 / 指标</th>' + ''.join('<th>' + html.escape(n) + '</th>' for n in order) + '</tr>')
        for m in MODALITIES:
            for metric in METRICS:
                parts.append('<tr><td>' + labels[m] + ' / ' + metric + '</td>' + ''.join('<td>' + format(reports[n][field][m][metric], '.6f') + '</td>' for n in order) + '</tr>')
        parts.append('</table>')
    parts.append('<h2>Holdout 样本与生成块数</h2><table><tr><th>模态</th><th>目标数</th><th>生成块数</th><th>噪声评估数</th></tr>')
    for m in MODALITIES:
        r = reports['anew_original']['results'][m]
        parts.append(f'<tr><td>{labels[m]}</td><td>{r["records"]}</td><td>{r["generated_blocks"]}</td><td>{r["noise_evaluations"]}</td></tr>')
    parts.append('</table><p>配对差异和置信区间见 COMPARISON.json；各模型原始记录、checkpoint 哈希和 W&amp;B 同步状态保存在 results 下。</p>')
    (ROOT / 'COMPARISON.html').write_text('\n'.join(parts), encoding='utf-8')
