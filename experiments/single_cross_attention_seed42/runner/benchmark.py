"""Run a supplied Omni adapter checkpoint through the locked trimodal benchmarks."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import socket

HERE = Path(__file__).resolve().parent
BASE_SHA = '961599acb617b8baf742755b4416322979507c9814ebd1a1e8582cff9c0a74c1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=HERE / 'runtime.json')
    parser.add_argument('--data-dir', type=Path, default=HERE.parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmarks', nargs='+', choices=['lnr', 'rabd', 'crossdocked2020'], default=['lnr', 'rabd', 'crossdocked2020'])
    parser.add_argument('--samples', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=20260814)
    parser.add_argument('--smoke', action='store_true', help='First target per modality, three fixed attempt slots by default')
    parser.add_argument('--stage', choices=['check', 'features', 'generate', 'score', 'all'], default='all')
    parser.add_argument('--wandb-project', default='anewomni-benchmarks')
    parser.add_argument('--wandb-entity')
    args = parser.parse_args()
    samples = args.samples if args.samples is not None else (3 if args.smoke else 30)
    if samples < 1 or not 1 <= args.batch_size <= samples:
        parser.error('Require samples >= batch-size >= 1')
    if len(args.benchmarks) != len(set(args.benchmarks)):
        parser.error('Benchmark names must be unique')
    checkpoint, data, out = args.checkpoint.resolve(), args.data_dir.resolve(), args.output.resolve()
    runtime = json.loads(args.runtime.resolve().read_text())
    for key in ['generation_python', 'donor_python', 'score_python', 'anew_source', 'boltzgen_source',
                'pepglad_source', 'cbg_source', 'base_checkpoint', 'boltzgen_assets']:
        candidate = Path(runtime[key]).expanduser()
        runtime[key] = str(candidate.absolute() if key.endswith('_python') else candidate.resolve())
        if not Path(runtime[key]).exists():
            raise FileNotFoundError(f'{key}: {runtime[key]}; run setup_runtime.py first')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    assert sha(runtime['base_checkpoint']) == BASE_SHA, 'Wrong fixed Anew base checkpoint'
    index_path = data / 'benchmarks/BENCHMARK_INDEX.json'
    index = json.loads(index_path.read_text())
    members = {m['path']: m for archive in index['archives'] for m in archive['members']}
    for item in index['metadata_files']:
        assert sha(data / item['path']) == item['sha256'], item['path']
    out.mkdir(parents=True, exist_ok=True)
    # One process owns a run directory. Multiple checkpoints use separate output directories.
    lock = out / '.active-run'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Run directory is in use: {lock}. If interrupted, verify its recorded process/job has exited before removing this lock directory.')
    write(lock / 'owner.json', {'pid': os.getpid(), 'host': socket.gethostname(),
                                'slurm_job_id': os.getenv('SLURM_JOB_ID')})
    def release_lock():
        (lock / 'owner.json').unlink()
        lock.rmdir()
    atexit.register(release_lock)
    mapping = [json.loads(x) for x in (data / 'benchmarks/SAMPLE_INPUT_MAP.jsonl').read_text().splitlines()]
    original = json.loads((data / 'benchmarks/provenance/feature_rows.json').read_text())
    original = {x['key']: x for x in original}
    specs, selected = [], []
    for name in args.benchmarks:
        path = data / 'benchmarks' / name / 'manifest.jsonl'
        targets = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(targets) == index['target_counts'][name]
        if args.smoke:
            targets = targets[:1]
        ids = {t['id'] for t in targets}
        selected += [x for x in mapping if x['benchmark'] == name and x['target'] in ids]
        for row in targets:
            for key in index['reference_path_fields']:
                if row.get(key):
                    path = data / row[key]
                    assert sha(path) == members[row[key]]['sha256'], row[key]
                    row[key] = str(path)
        path = out / 'manifests' / name / 'manifest.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ''.join(json.dumps(x) + '\n' for x in targets)
        if path.exists():
            assert path.read_text() == content, 'Output directory belongs to another target selection'
        else:
            path.write_text(content)
        specs.append({'name': name, 'targets': len(targets), 'manifest': str(path), 'manifest_sha256': sha(path)})
    config = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'data_dir': str(data),
              'output': str(out), 'cache': str(out / 'cache'), 'runtime': runtime,
              'base_checkpoint_sha256': BASE_SHA, 'dataset_index_sha256': sha(index_path),
              'benchmark_specs': specs, 'samples': samples, 'batch_size': args.batch_size,
              'gamma': args.gamma, 'seed': args.seed, 'smoke': args.smoke,
              'wandb_project': args.wandb_project, 'wandb_entity': args.wandb_entity,
              'source_hashes': {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob('*.py') if 'vendor' not in p.parts}}
    env = {k: v for k, v in os.environ.items() if k not in ['PYTHONPATH', 'PYTHONHOME', '_WANDB_SERVICE', 'WANDB_SERVICE']}
    config_path = out / 'run.json'
    env.update(OMNI_BENCH_CONFIG=str(config_path), OMP_NUM_THREADS='4', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def call(stage, script, *extra):
        stage_env = env.copy()
        paths = [str(HERE)] + runtime.get(stage + '_extra_pythonpath', [])
        paths += ([runtime['anew_source']] if stage == 'generation' else
                  [str(Path(runtime['boltzgen_source']) / 'src')] if stage == 'donor' else
                  [runtime['anew_source'], runtime['cbg_source']])
        stage_env['PYTHONPATH'] = os.pathsep.join(paths)
        python = runtime[stage + '_python']
        stage_env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
        print('STAGE', stage, script, flush=True)
        subprocess.run([python, '-u', str(HERE / script), *extra], env=stage_env, cwd=out, check=True)

    # The temporary config is used only for schema/environment inspection.
    old_config = json.loads(config_path.read_text()) if config_path.exists() else None
    temporary = out / 'preflight.json'
    write(temporary, config)
    env['OMNI_BENCH_CONFIG'] = str(temporary)
    call('generation', 'inspect_checkpoint.py')
    inspected = json.loads((out / 'CHECKPOINT_CHECK.json').read_text())
    config.update(variant=inspected['variant'], projection_width=inspected['projection_width'])
    config['run_hash'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if old_config is not None:
        assert old_config == config, 'Checkpoint, source, runtime or settings changed; use a new output directory'
    else:
        write(config_path, config)
    env['OMNI_BENCH_CONFIG'] = str(config_path)
    for stage in ['generation', 'score'] + (['donor'] if config['variant'] == 'full' else []):
        call(stage, 'check_environment.py', stage)
    feature_rows = []
    for row in selected:
        key = row['key']
        cached = original[key]
        for path, digest in cached['files'].items():
            src = data / 'benchmarks/prepared' / path
            assert sha(src) == digest, path
            dest = out / 'cache' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(src, dest)
            assert sha(dest) == digest
        for filename in ['metadata.json', 'target.cif']:
            rel = f'benchmarks/prepared/inputs/{key}/{filename}'
            src = data / rel
            assert sha(src) == members[rel]['sha256']
            dest = out / 'cache/inputs' / key / filename
            if not dest.exists():
                shutil.copyfile(src, dest)
        feature_rows.append(cached)
    write(out / 'cache/manifests/feature_rows.json', feature_rows)
    write(out / 'PREFLIGHT.json', {'passed': True, 'targets': len(selected), 'run_hash': config['run_hash'],
                                 'checkpoint': inspected, 'gpu_generation_tested': False})
    if args.stage == 'check':
        print('PREFLIGHT_PASSED; GPU generation/scoring are not yet verified', flush=True)
        return
    if args.stage in ['all', 'features'] and config['variant'] == 'full':
        call('donor', 'extract_features.py', '--shard', '0', '--shards', '1')
    if args.stage in ['all', 'generate']:
        call('generation', 'generate.py')
    if args.stage in ['all', 'score']:
        call('score', 'score.py')
    print('FINISHED_STAGE', args.stage, str(out), flush=True)


if __name__ == '__main__':
    main()
