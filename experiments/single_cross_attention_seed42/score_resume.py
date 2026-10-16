"""Evaluate fixed slots with the pinned scorers and explicit metric coverage."""
import collections
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from runtime import CONFIG, ROOT, HERE, atomic_json, sha, Tracker


def numeric_fields(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in {'n', 'version', 'errors', 'target_id', 'atom_num'}:
                yield from numeric_fields(item, prefix + '/' + key if prefix else key)
    elif isinstance(value, (int, float)) and math.isfinite(value):
        yield prefix, float(value)


def aggregate(events):
    values = collections.defaultdict(list)
    for event in events:
        for key, value in numeric_fields(event.get('scores')):
            values[key].append(value)
    return {'attempted': len(events), 'generated': sum(e['success'] for e in events),
            'metric_error_candidates': sum(bool((e.get('scores') or {}).get('errors')) for e in events),
            'metrics': {k: {'count': len(v), 'mean': statistics.mean(v), 'median': statistics.median(v)}
                        for k, v in sorted(values.items())}}


def score_target(spec,index,row):
    name = spec['name']
    target = row['id']
    generation = ROOT / 'generation/omni' / name
    slots = [json.loads((generation / 'slots' / target / f'{n:02d}.json').read_text())
             for n in range(CONFIG['samples'])]
    assert all(x['id'] == target and x['n'] == n for n, x in enumerate(slots))
    success = [x for x in slots if x['success']]
    for record in success:
        assert sha(generation / target / f"{record['n']}.pdb") == record['pdb_sha256']
    previous = ROOT/'scores'/name/target/'SCORING_COMPLETE.json'
    if previous.exists():
        report = json.loads(previous.read_text())
        assert report['run_hash'] == CONFIG['run_hash'] and report['checkpoint_sha256'] == CONFIG['checkpoint_sha256']
        for filename,digest in report['score_files'].items():
            assert sha(previous.parent/filename) == digest
        events=[json.loads(x) for x in (previous.parent/'evaluation.jsonl').read_text().splitlines()]
        assert len(events)==CONFIG['samples'] and all(x['target']==target and x['n']==n for n,x in enumerate(events))
        return report,events
    cache = ROOT / 'scores' / name
    work = cache / target
    work.mkdir(parents=True, exist_ok=True)
    inputs = work / 'candidate_input'
    inputs.mkdir(exist_ok=True)
    if success:
        link = inputs / target
        if not link.exists():
            link.symlink_to(generation / target, target_is_directory=True)
        assert link.resolve() == (generation / target).resolve()
    (inputs / 'results.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in success))
    if success:
        small = name == 'crossdocked2020'
        script = HERE / 'scoring' / ('evaluate_small_molecule_target.py' if small else 'evaluate_biologic_target.py')
        command = [sys.executable, str(script), '--manifest', spec['manifest'],
                   '--candidates', str(inputs), '--cache-root', str(cache),
                   '--target-index', str(index), '--n-samples', str(CONFIG['samples'])]
        command += (['--cbg-root', CONFIG['runtime']['cbg_source'], '--exhaustiveness', '16'] if small
                    else ['--pepglad-root', CONFIG['runtime']['pepglad_source'],
                          '--mode', 'antibody' if name == 'rabd' else 'peptide'])
        with (work / 'scoring.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(work / 'SCORING_ERROR.json', {'returncode': result.returncode, 'command': command})
            raise RuntimeError('Scorer failed; inspect ' + str(work / 'scoring.log'))
        summary = json.loads((work / 'target_summary.json').read_text())
        assert summary['requested'] == CONFIG['samples'] and summary['generation_successes'] == len(success)
    else:
        summary = {'requested': CONFIG['samples'], 'generation_successes': 0, 'evaluated': 0,
                   'reason': 'all_predeclared_generation_slots_failed'}
        atomic_json(work / 'target_summary.json', summary)
    events = []
    for record in slots:
        candidate = work / f"candidate_{record['n']:03d}.json"
        events.append({'benchmark': name, 'target': target, 'n': record['n'],
                       'success': record['success'], 'failure_stage': record.get('failure_stage'),
                       'scores': json.loads(candidate.read_text()) if candidate.is_file() else None})
    (work / 'evaluation.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in events))
    report = {'benchmark': name, 'target': target, 'summary': summary,
              'checkpoint_sha256': CONFIG['checkpoint_sha256'], 'run_hash': CONFIG['run_hash'],
              'score_files': {p.name: sha(p) for p in work.glob('candidate_*.json')}}
    atomic_json(work / 'SCORING_COMPLETE.json', report)
    return report, events


def main():
    import os
    early = os.environ.get('BENCH_EARLY_SCORE') == '1'
    tracker = Tracker('scoring-early' if early else 'scoring', {
        'scoring_executor_sha256': sha(Path(__file__)), 'early_cpu_overlap': early})
    started = time.monotonic()
    reports, all_events = [], []
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    pending = [(spec,index,row) for spec in CONFIG['benchmark_specs']
               for index,row in enumerate([json.loads(line) for line in Path(spec['manifest']).read_text().splitlines()])]
    workers = int(os.environ.get('BENCH_SCORE_WORKERS','24'))
    deadline = time.monotonic() + float(os.environ.get('BENCH_SCORE_WINDOW_SECONDS', 3.4*3600))
    idle_seconds = float(os.environ.get('BENCH_SCORE_IDLE_SECONDS', 120))
    idle_since = time.monotonic()

    def finish_partial(reason):
        summary = {'benchmark_complete': False, 'pause_reason': reason,
                   'targets_scored': len(reports), 'pending_targets': len(pending),
                   'run_hash': CONFIG['run_hash'], 'checkpoint_sha256': CONFIG['checkpoint_sha256'],
                   'seconds': time.monotonic()-started, 'early_cpu_overlap': early}
        tracker.finish(summary)
        atomic_json(ROOT/('EARLY_SCORING_COMPLETE.json' if early else 'SCORING_PAUSED.json'), summary)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = {}
        while pending or running:
            if time.monotonic() >= deadline and not running:
                finish_partial('CPU allocation window ending; verified scores retained for resume')
                return
            for item in list(pending) if time.monotonic() < deadline else []:
                spec,index,row = item
                slots = ROOT/'generation/omni'/spec['name']/'slots'/row['id']
                if len(running) < workers and all((slots/f'{n:02d}.json').is_file() for n in range(CONFIG['samples'])):
                    running[pool.submit(score_target,*item)] = item
                    pending.remove(item)
            if running:
                idle_since = time.monotonic()
                done,_ = wait(running,timeout=3,return_when=FIRST_COMPLETED)
                for future in done:
                    item = running.pop(future)
                    try:
                        report,events = future.result()
                    except Exception as exc:
                        atomic_json(ROOT/'SCORING_FAILURE.json',{'target':item[2]['id'],'error':repr(exc)})
                        raise
                    reports.append(report); all_events.extend(events)
                    tracker.log({'targets_scored':len(reports),'attempted':len(all_events),
                                 'generated':sum(x['success'] for x in all_events)})
                    atomic_json(ROOT/'PARTIAL_REPORT.json',{'targets_scored':len(reports),
                        'checkpoint_sha256':CONFIG['checkpoint_sha256'],
                        'benchmark_results':{spec['name']:aggregate([x for x in all_events if x['benchmark']==spec['name']]) for spec in CONFIG['benchmark_specs']}})
            else:
                if time.monotonic()-idle_since > idle_seconds:
                    finish_partial('No additional generated targets ready; release allocation and resume when slots arrive')
                    return
                time.sleep(3)

    if early:
        finish_partial('All target scores cached; final allocation must verify complete generation and publish the audited report')
        return
    overlap = json.loads((Path(CONFIG['data_dir']) / 'benchmarks/provenance/PDB_OVERLAP_AUDIT.json').read_text())
    overlap_ids = {b['benchmark']: {x['target'] for x in b['matches']} for b in overlap['benchmarks']}
    by_benchmark, verified = {}, {}
    for spec in CONFIG['benchmark_specs']:
        name = spec['name']
        events = [x for x in all_events if x['benchmark'] == name]
        by_benchmark[name] = {'all': aggregate(events),
            'pdb_overlap': aggregate([x for x in events if x['target'] in overlap_ids[name]]),
            'no_detected_pdb_overlap': aggregate([x for x in events if x['target'] not in overlap_ids[name]])}
        summaries = [x['summary'] for x in reports if x['benchmark'] == name]
        required = (['complete_core', 'complete_vina', 'complete_interaction',
                     'reference_vina_complete', 'reference_interaction_complete']
                    if name == 'crossdocked2020' else ['structural_complete', 'rosetta_complete'])
        verified[name] = all(sum(s.get(k, 0) for s in summaries) > 0 for k in required)
    result = {'checkpoint_sha256': CONFIG['checkpoint_sha256'], 'run_hash': CONFIG['run_hash'], 'benchmark_complete': True,
              'variant': CONFIG['variant'], 'benchmark_results': by_benchmark,
              'real_generation_and_full_scoring_verified': verified,
              'smoke': CONFIG['smoke'], 'seconds': time.monotonic() - started,
              'policy': 'Fixed attempted slots; no success-based resampling. Each metric reports its own finite-value denominator.'}
    atomic_json(ROOT / 'REPORT.json', result)
    tracker.finish(result, [ROOT / 'REPORT.json', ROOT / 'run.json'])
    if CONFIG['smoke'] and not all(verified.values()):
        raise RuntimeError('Smoke did not verify real generation plus full scoring in every selected modality; inspect REPORT.json')
    atomic_json(ROOT / 'COMPLETE.json', {'run_hash': CONFIG['run_hash'], 'report_sha256': sha(ROOT / 'REPORT.json'),
                                       'real_generation_and_full_scoring_verified': verified})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
