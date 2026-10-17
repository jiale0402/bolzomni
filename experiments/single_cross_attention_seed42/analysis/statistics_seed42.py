"""Post-hoc target-cluster inference for all 27 published seed-1 endpoints.

No model fitting, scoring or generation. Candidate groups and missing values
travel together in resampling, preserving the published finite-score estimand.
"""
import argparse
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import tarfile

import numpy as np

BASE = Path('output/single_cross_attention_20261015')
SEED = 1
OUT = BASE / 'statistics_seed1'
BOOTSTRAPS = 50000
PERMUTATIONS = 200000
SPECS = [
    ('rabd', ['structural/complex_RMSD', 'structural/AAR',
              'structural/intra_region_clash', 'structural/interface_clash',
              'dG', 'ddG', 'DockQ', 'iRMSD', 'fnat', 'backbone_interface_lddt'],
     [1, 100, 100, 100, 1, 1, 1, 1, 100, 1]),
    ('lnr', ['structural/complex_RMSD', 'structural/peptide_RMSD', 'structural/AAR',
             'structural/intra_region_clash', 'structural/interface_clash',
             'dG', 'ddG', 'DockQ', 'iRMSD', 'fnat', 'backbone_interface_lddt'],
     [1, 1, 100, 100, 100, 1, 1, 1, 1, 100, 1]),
    ('crossdocked2020', ['vina/score_only', 'vina/minimize', 'vina/dock',
                        'chemical/qed', 'chemical/sa', 'chemical/lipinski'], [1]*6),
]


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def key(row):
    return row['benchmark'], row['target'], row['n']


def nested(scores, name):
    value = scores
    for part in name.split('/'):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(order))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., pvalues[index]*(len(order)-rank)))
        adjusted[index] = running
    return adjusted.tolist()


def cluster_statistics(a, b, seed):
    # Each row is a target; each column one of its original 30 slots.
    sums = [np.nansum(x, axis=1) for x in (a, b)]
    counts = [np.isfinite(x).sum(axis=1) for x in (a, b)]
    means = [float(s.sum()/n.sum()) for s, n in zip(sums, counts)]
    observed = means[0]-means[1]  # method minus baseline
    rng = np.random.default_rng(seed)
    boots = []
    extreme = 0
    tolerance = max(abs(observed)*1e-12, 1e-14)
    for start in range(0, BOOTSTRAPS, 1000):
        indices = rng.integers(0, len(a), (min(1000, BOOTSTRAPS-start), len(a)))
        den = [c[indices].sum(axis=1) for c in counts]
        assert all(np.all(c > 0) for c in den), 'Undefined bootstrap replicate'
        boots.extend(sums[0][indices].sum(axis=1)/den[0] - sums[1][indices].sum(axis=1)/den[1])
    # Swap complete model-labelled target groups, including missingness. For
    # equal per-target counts this is exactly a target-difference sign flip.
    for start in range(0, PERMUTATIONS, 2000):
        swap = rng.integers(0, 2, (min(2000, PERMUTATIONS-start), len(a)), dtype=np.int8)
        ps = sums[0] + swap*(sums[1]-sums[0])
        pn = counts[0] + swap*(counts[1]-counts[0])
        qs = sums[1] + swap*(sums[0]-sums[1])
        qn = counts[1] + swap*(counts[0]-counts[1])
        assert np.all(pn.sum(1)>0) and np.all(qn.sum(1)>0)
        null = ps.sum(1)/pn.sum(1) - qs.sum(1)/qn.sum(1)
        extreme += int((np.abs(null) >= abs(observed)-tolerance).sum())
    return dict(method_mean=means[0], baseline_mean=means[1],
                difference_method_minus_baseline=observed,
                paired_target_bootstrap_ci95=np.quantile(boots, [.025, .975]).tolist(),
                paired_target_swap_two_sided_p=(extreme+1)/(PERMUTATIONS+1),
                finite_candidates=[int(c.sum()) for c in counts],
                valid_targets=[int((c>0).sum()) for c in counts],
                paired_valid_targets=int(((counts[0]>0)&(counts[1]>0)).sum()),
                attempted_slots=int(a.size), targets=len(a))


def main():
    OUT.mkdir(exist_ok=True)
    table_path = BASE/'main_table_preview/MAIN_TABLE.json'
    table = read(table_path)
    native_path = BASE/'statistics_seed1/method_native_inputs.json.gz'
    native = json.loads(gzip.decompress(native_path.read_bytes()))
    method = {key(r): r for r in native['records']}
    baseline = {}
    archive = BASE/f'seed{SEED}_evidence/audits__final_score_evidence.tar.gz'
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.endswith('/evaluation.jsonl'):
                for line in tar.extractfile(member).read().splitlines():
                    row = json.loads(line)
                    row['metrics'] = {name: nested(row['scores'], name)
                                      for _, names, _ in SPECS for name in names}
                    assert key(row) not in baseline
                    baseline[key(row)] = row
    assert set(method) == set(baseline) and len(method) == 7590
    paths = [native_path, archive, table_path]
    baseline_interface = BASE/f'seed{SEED}_evidence/interface__candidates.jsonl'
    interface_summary = read(BASE/f'seed{SEED}_evidence/interface__SUMMARY.json')
    assert sha(baseline_interface) == interface_summary['candidate_sha256']
    extra = Path('output/trimodal_extended_metrics_20261011/candidates.jsonl')
    more = Path('output/trimodal_more_metrics_20261011/ADDITIONAL_CANDIDATES.jsonl')
    assert sha(extra) == read(extra.parent/'SCORING_COMPLETE.json')['candidate_jsonl_sha256']
    more_summary = read(more.parent/'SUMMARY.json')
    assert sha(more) == more_summary['new_scoring']['candidate_sha256']
    assert more_summary['checkpoints']['frozen4m']['checkpoint_sha256'] == native['checkpoint_sha256']
    for path, dest, model in [(baseline_interface, baseline, None),
                              (extra, method, 'frozen4m'), (more, method, 'frozen4m')]:
        paths.append(path)
        for row in records(path):
            if model and row['model'] != model:
                continue
            if row['benchmark'] not in ('lnr', 'rabd'):
                continue
            target = dest[key(row)]
            if model:
                assert row['candidate_pdb_sha256'] == target['candidate_pdb_sha256'], key(row)
            for name in ('DockQ', 'iRMSD', 'fnat', 'backbone_interface_lddt'):
                if name in row['metrics']:
                    target['metrics'][name] = row['metrics'][name]
    results = []
    for (bench, names, scales), tab in zip(SPECS, table['tables']):
        assert bench == tab['benchmark']
        targets = sorted({k[1] for k in method if k[0] == bench})
        published = [next(r for r in tab['rows'] if r['method']==label)['values']
                     for label in ('Our method', f'Single cross-attention (seed {SEED})')]
        for index, (metric, scale, up) in enumerate(zip(names, scales, tab['higher_is_better'])):
            arrays = []
            for model in (method, baseline):
                a = np.full((len(targets), 30), np.nan)
                for i, t in enumerate(targets):
                    for n in range(30):
                        v = model[(bench,t,n)]['metrics'].get(metric)
                        if finite(v):
                            a[i,n] = v*scale
                arrays.append(a)
            seed = int.from_bytes(hashlib.sha256((bench+'/'+metric+f'/seed{SEED}').encode()).digest()[:8], 'big')
            r = cluster_statistics(*arrays, seed)
            for value, reported in zip([r['method_mean'],r['baseline_mean']], published):
                assert f'{value:.4f}' == reported[index], (bench, metric, value, reported[index])
            r.update(benchmark=bench, metric=metric, name=tab['columns'][index],
                     higher_is_better=up, method_better=r['difference_method_minus_baseline']*(1 if up else -1)>0,
                     random_seed=seed, target_ids=targets)
            results.append(r)
            print(bench, metric, r['paired_target_swap_two_sided_p'], flush=True)
    assert len(results)==27
    for r, adjusted in zip(results, holm([r['paired_target_swap_two_sided_p'] for r in results])):
        r['holm_adjusted_p_27'] = adjusted
        r['conclusion'] = ('method_better' if r['method_better'] else 'baseline_better') if adjusted < .05 else 'not_significant'
    report = dict(scope=f'All 27 main-table metrics, method vs single cross-attention training seed {SEED}; post-hoc exploratory analysis.',
                  family_size=27, bootstrap_replicates=BOOTSTRAPS, permutation_replicates=PERMUTATIONS,
                  ci='Pointwise unadjusted paired target-cluster percentile 95% bootstrap intervals; do not interpret as Holm-adjusted intervals.',
                  test='Two-sided random target-group label swaps, absolute difference statistic, plus-one Monte Carlo correction; null assumes exchangeability of model-labelled target groups.',
                  estimand='Published finite-candidate mean. Protein scores have 30 valid scores per usable target; ligand scores pool finite candidates with differing model coverage. Entire 30-slot target groups and missingness are retained.',
                  correction='Holm step-down family-wise correction over these 27 endpoints; excludes unplanned historical model-development choices.',
                  limitations=[f'One historical method seed vs baseline seed {SEED}; no training-seed uncertainty.', ('Different A100 variants at equal wall-clock budget.' if SEED==1 else 'Same A100-PCIE-40GB model and wall-clock budget; different optimizer steps/presentations and training seeds.'), 'Benchmark previously inspected; 19 ligand training-PDB overlaps.', 'Missing scores are not imputed; ligand inference conditional on observed finite scores.', 'Target independence/exchangeability does not establish independence of homologous complexes.'],
                  checkpoints=dict(method=native['checkpoint_sha256'], baseline=interface_summary['checkpoint_sha256']),
                  input_hashes={str(p):sha(p) for p in paths}, source_sha256=sha(Path(__file__)), results=results)
    (OUT/'ALL_METRICS_STATISTICS.json').write_text(json.dumps(report,indent=2)+'\n')
    body = ['<!doctype html><meta charset="utf-8"><title>All 27 metric statistics</title>',
            '<style>body{font:15px Arial;margin:30px;color:#162437}table{border-collapse:collapse;width:100%;margin-bottom:30px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}u{text-underline-offset:3px}.yes{background:#e5f5ee}</style>',
            f'<h1>Method vs single cross-attention baseline, seed {SEED}</h1>',
            '<p>27 endpoints; 50,000 target bootstraps, 200,000 paired target-group swaps. Difference = method − baseline. CI is pointwise; Holm p adjusts all 27 endpoints. Bold/underline reflect displayed mean rank only.</p>']
    for bench, _, _ in SPECS:
        body += [f'<h2>{bench}</h2><table><tr><th>Metric</th><th>Method</th><th>Baseline</th><th>Difference [95% CI]</th><th>p</th><th>Holm p</th><th>Conclusion</th><th>Finite candidates M/B</th><th>Targets M/B</th></tr>']
        for r in results:
            if r['benchmark']!=bench:
                continue
            a,b=(f'{r[x]:.4f}' for x in ['method_mean','baseline_mean'])
            if a==b:
                a,b=f'<b>{a}</b>',f'<b>{b}</b>'
            elif r['method_better']:
                a,b=f'<b>{a}</b>',f'<u>{b}</u>'
            else:
                a,b=f'<u>{a}</u>',f'<b>{b}</b>'
            ci=r['paired_target_bootstrap_ci95']
            cells=[html.escape(r['name'])+(' ↑' if r['higher_is_better'] else ' ↓'),a,b,
                   f"{r['difference_method_minus_baseline']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]",
                   f"{r['paired_target_swap_two_sided_p']:.6g}",f"{r['holm_adjusted_p_27']:.6g}",r['conclusion'],
                   '/'.join(map(str,r['finite_candidates'])),'/'.join(map(str,r['valid_targets']))]
            body.append(('<tr class="yes">' if r['conclusion']!='not_significant' else '<tr>')+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
        body.append('</table>')
    body.append('<p>'+html.escape(' '.join(report['limitations']))+'</p>')
    (OUT/'ALL_METRICS_STATISTICS.html').write_text(''.join(body),encoding='utf-8')
    print(json.dumps({'significant_after_Holm':[(r['benchmark'],r['metric'],r['conclusion']) for r in results if r['conclusion']!='not_significant']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[1,42],default=1)
    SEED=parser.parse_args().seed;OUT=BASE/f'statistics_seed{SEED}'
    assert holm([.01,.04,.03]) == [.03,.06,.06]
    main()
