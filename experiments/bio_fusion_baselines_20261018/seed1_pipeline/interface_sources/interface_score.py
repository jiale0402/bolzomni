"""Score preserved protein slots with the exact main-table interface operators."""
import concurrent.futures as cf
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
from common import ROOT, CONFIG, Tracker, atomic_json, sha
import evaluate as extended
import score_more as more

METRICS = ('DockQ', 'iRMSD', 'fnat', 'backbone_interface_lddt')


def score_target(row):
    start = time.monotonic()
    directory = Path(CONFIG['benchmark_root']) / 'scores' / row['benchmark'] / row['id']
    candidates = {x['n']: x for x in extended.jsonl(directory / 'candidate_input/results.jsonl')}
    old = {x['n']: x for x in extended.jsonl(directory / 'evaluation.jsonl')}
    assert set(old) == set(range(30))
    inputs = {'reference': sha(row['reference_pdb']), 'evaluation': sha(directory / 'evaluation.jsonl'),
              'candidates': sha(directory / 'candidate_input/results.jsonl')}
    for n, candidate in candidates.items():
        assert sha(directory / 'candidate_input' / row['id'] / f'{n}.pdb') == candidate['pdb_sha256']
    out = ROOT / 'targets' / row['benchmark'] / (row['id'] + '.json')
    if out.exists():
        saved = json.loads(out.read_text())
        assert saved['input_hashes'] == inputs and saved['protocol_sha256'] == sha(ROOT / 'code/protocol.json')
        assert len(saved['records']) == 30
        return saved
    reference = more.mapped(row['reference_pdb'], row, more.native.tokenize_sequence(row['reference_sequence']))
    records = []
    first_success = min(candidates, default=-1)
    with tempfile.TemporaryDirectory(prefix='single-ca-interface-', dir=os.getenv('TMPDIR')) as scratch:
        for n in range(30):
            record = dict(model='enzy_transfer', training_seed=CONFIG['training_seed'], benchmark=row['benchmark'], target=row['id'], n=n)
            if not old[n]['success']:
                assert n not in candidates
                record.update(metrics={k: None for k in METRICS}, errors={'generation': old[n].get('failure_stage')})
                records.append(record)
                continue
            c = candidates[n]
            assert c['success'] and old[n]['success']
            pdb = directory / 'candidate_input' / row['id'] / f'{n}.pdb'
            record.update(extended.score_candidate(row, c, pdb, old[n], scratch, self_test=(row['index'] == 0 and n == first_success)))
            generated = more.mapped(pdb, row, more.native.tokenize_sequence(c['gen_seq']))
            rt, rl = reference
            gt, gl = generated
            value, denominator = more.lddt(more.atoms(rl), more.atoms(gl), more.atoms(rt), more.atoms(gt))
            record['metrics']['backbone_interface_lddt'] = value
            record['backbone_interface_distance_pairs'] = denominator
            if n == first_success:
                identity, count = more.lddt(more.atoms(rl), more.atoms(rl), more.atoms(rt), more.atoms(rt))
                assert identity is None or abs(identity - 1.) < 1e-12
                record['backbone_identity_self_test'] = dict(value=identity, pairs=count)
            records.append(record)
    result = dict(benchmark=row['benchmark'], target=row['id'], records=records,
                  input_hashes=inputs, protocol_sha256=sha(ROOT / 'code/protocol.json'), elapsed_seconds=time.monotonic()-start)
    atomic_json(out, extended.clean(result))
    return result


def main():
    start = time.monotonic()
    assert importlib.metadata.version('DockQ') == '2.1.3'
    for name, digest in json.loads((ROOT / 'DEPLOYED.json').read_text()).items():
        assert sha(ROOT / 'code' / name) == digest, name
    b = Path(CONFIG['benchmark_root'])
    cfg = json.loads((b / 'run.json').read_text())
    assert cfg['run_hash'] == CONFIG['benchmark_run_hash']
    assert sha(cfg['checkpoint']) == cfg['checkpoint_sha256'] == CONFIG['checkpoint_sha256']
    done = json.loads((b / 'COMPLETE.json').read_text())
    assert done['run_hash'] == cfg['run_hash'] and sha(b / 'REPORT.json') == done['report_sha256']
    rows = []
    for spec in cfg['benchmark_specs']:
        if spec['name'] in ('lnr', 'rabd'):
            assert sha(spec['manifest']) == spec['manifest_sha256']
            rows.extend(extended.jsonl(spec['manifest']))
    assert len(rows) == 153
    tracker = Tracker('remaining-interface', {'protocol_sha256': sha(ROOT / 'code/protocol.json'), 'new_generation_attempts': 0})
    try:
        results = []
        workers = int(os.getenv('INTERFACE_WORKERS', '8'))
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_target, row) for row in rows]
            for future in cf.as_completed(futures):
                result = future.result()
                results.append(result)
                tracker.log(dict(targets_done=len(results), target=result['target'], benchmark=result['benchmark'], candidates=30))
                atomic_json(ROOT / 'PROGRESS.json', dict(targets_done=len(results), targets_expected=153, elapsed_seconds=time.monotonic()-start))
        records = [r for t in sorted(results, key=lambda t: (t['benchmark'], t['target'])) for r in t['records']]
        assert len(records) == 4590 and len({(r['benchmark'], r['target'], r['n']) for r in records}) == 4590
        (ROOT / 'candidates.jsonl').write_text(''.join(json.dumps(extended.clean(r), allow_nan=False)+'\n' for r in records))
        summary = dict(checkpoint_sha256=CONFIG['checkpoint_sha256'], training_seed=CONFIG['training_seed'],
                       benchmark_run_hash=cfg['run_hash'], targets=153, attempted_slots=4590, allocated_cpus=workers,
                       elapsed_seconds=time.monotonic()-start, new_generation_attempts=0, optimizer_updates=0,
                       candidate_sha256=sha(ROOT / 'candidates.jsonl'), metrics={})
        for bench, expected in [('lnr', 2790), ('rabd', 1800)]:
            summary['metrics'][bench] = {}
            for metric in METRICS:
                target_values = []
                count = 0
                for result in results:
                    if result['benchmark'] != bench:
                        continue
                    values = [r['metrics'][metric] for r in result['records'] if r['metrics'].get(metric) is not None and math.isfinite(r['metrics'][metric])]
                    count += len(values)
                    if values:
                        target_values.append(float(np.mean(values)))
                summary['metrics'][bench][metric] = dict(mean=float(np.mean(target_values)) if target_values else None,
                    targets=len(target_values), candidates=count, attempted_slots=expected, missing=expected-count)
        atomic_json(ROOT / 'SUMMARY.json', summary)
        tracker.finish(summary, [ROOT / 'SUMMARY.json', ROOT / 'candidates.jsonl', ROOT / 'DEPLOYED.json', ROOT / 'code/protocol.json'])
        atomic_json(ROOT / 'COMPLETE.json', summary)
        print(json.dumps(summary), flush=True)
    except Exception:
        atomic_json(ROOT / 'FAILED.json', {'traceback': traceback.format_exc()})
        tracker.run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
