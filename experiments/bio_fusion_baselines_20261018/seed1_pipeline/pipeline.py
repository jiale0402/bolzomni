"""Atomic progression with per-stage submission ownership."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def submit(stage, args, script):
    assert not any((p/'STOPPED.json').exists() for p in [ROOT, ROOT.parent]), 'Run is stopped'
    receipt = ROOT/(stage+'_SUBMISSION.json')
    if receipt.exists():
        return read(receipt)['stdout'].strip()
    method = read(ROOT/'code/config.json')['method_id']
    cfg = read(ROOT/'code/config.json')
    assert cfg.get('training_enabled',False) and cfg['trainable_parameters'] == 4027715
    name = f"bio-{method}-seed{cfg['seed']}-{stage}"
    # Recover a successful submission recorded by the submitter if a previous
    # client died before writing the stage receipt. Never blind-submit twice.
    journal = ROOT/'submissions.jsonl'
    if journal.exists():
        hits = [json.loads(line) for line in journal.read_text().splitlines()]
        hits = [r for r in hits if r['request']['name']==name and r['returncode']==0]
        if hits:
            assert len(hits)==1, 'Multiple successful submissions require investigation'
            write(receipt, hits[0])
            return hits[0]['stdout'].strip()
    intent = ROOT/(stage+'_SUBMISSION_INTENT')
    intent.mkdir(exist_ok=False)
    write(intent/'owner.json', dict(host=socket.gethostname(),pid=os.getpid(),unix=time.time(),stage=stage))
    result = subprocess.run(['python3', str(ROOT/'code/submit.py'), *args, '--name', name,
                             str(ROOT/script)], env=dict(os.environ, FUSION_ROOT=str(ROOT)),
                            capture_output=True, text=True)
    if result.returncode:
        write(ROOT/(stage+'_SUBMIT_FAILED.json'), dict(unix=time.time(), stdout=result.stdout, stderr=result.stderr))
        raise RuntimeError(result.stdout+result.stderr)
    record = json.loads(result.stdout)
    write(receipt, record)
    return record['stdout'].strip()


def terminal_state(job):
    p = subprocess.run(['sacct','-X','-n','-P','-j',job,'-o','State'],capture_output=True,text=True)
    assert p.returncode == 0,p.stderr
    rows = [x.split('|')[0].strip().split()[0] for x in p.stdout.splitlines() if x.strip()]
    return bool(rows) and all(x == 'COMPLETED' for x in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--advance', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if any((p/'STOPPED.json').exists() for p in [ROOT, ROOT.parent]):
        print(json.dumps(dict(stopped=True)))
        return
    mutex = ROOT/'progression_mutex'
    try:
        mutex.mkdir()
    except FileExistsError:
        print(json.dumps(dict(locked=True, owner=read(mutex/'owner.json') if (mutex/'owner.json').exists() else None)))
        return
    owner = dict(host=socket.gethostname(), pid=os.getpid(), unix=time.time(),slurm_job_id=os.getenv('SLURM_JOB_ID'))
    write(mutex/'owner.json', owner)
    try:
        cfg = read(ROOT/'code/config.json')
        status = dict(unix=time.time(), method=cfg['method_id'], root=str(ROOT),
            training_complete=(ROOT/'training/COMPLETE.json').exists(),
            benchmark_complete=(ROOT/'benchmark/COMPLETE.json').exists(),
            interface_complete=(ROOT/'interface_metrics/COMPLETE.json').exists(),
            comparison_complete=(ROOT/'comparison/COMPLETE.json').exists())
        jobs = {p.name.removesuffix('_SUBMISSION.json'): read(p)['stdout'].strip()
                for p in ROOT.glob('*_SUBMISSION.json')}
        status['jobs'] = jobs
        if jobs:
            p = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(jobs.values()),
                '-o', 'JobID,State,Elapsed,AllocTRES'], capture_output=True, text=True)
            status['slurm'] = p.stdout
        if args.advance:
            if not status['training_complete']:
                if 'train' not in jobs:
                    jobs['train'] = submit('train', ['--partition',os.environ.get('GPU_PARTITION','gpu'),'--time','04:15:00',
                        '--mem','64G','--cpus','4','--gres','gpu:a100:1','--constraint','pcie'], 'train.sbatch')
            else:
                subprocess.run([sys.executable, str(ROOT/'prepare_benchmark.py')], check=True, stdout=sys.stderr)
                if 'generation' not in jobs:
                    jobs['generation'] = submit('generation', ['--partition',os.environ.get('GPU_PARTITION','gpu'),'--time','04:00:00',
                        '--mem','64G','--cpus','4','--gres','gpu:a100:1','--array','0-3%4'], 'generate.sbatch')
                chunks = list((ROOT/'benchmark/generation/omni').glob('COMPLETE-chunk*.json'))
                status['generation_chunks_complete'] = len(chunks)
                if len(chunks)==31:
                    rows = [read(p) for p in chunks]
                    bench = read(ROOT/'benchmark/run.json')
                    assert all(r['provenance']['run_hash']==bench['run_hash'] and r['attempted']==r['expected'] for r in rows)
                    assert sum(r['attempted'] for r in rows)==7590
                    receipt = ROOT/'benchmark/generation/omni/COMPLETE-shard0.json'
                    if not receipt.exists():
                        write(receipt, dict(provenance=rows[0]['provenance'], attempted=7590, expected=7590,
                            successful=sum(r['successful'] for r in rows), targets=253, chunks=31,
                            fixed_attempts_per_target=30, generation_complete=True))
                    if not (ROOT/'benchmark/COMPLETE.json').exists():
                        stages = sorted(k for k in jobs if k.startswith('scoring'))
                        if not stages or terminal_state(jobs[stages[-1]]):
                            assert len(stages) < 6,'Inspect native scoring progress before further allocations'
                            stage = 'scoring'+str(len(stages))
                            jobs[stage] = submit(stage, ['--partition',os.environ.get('CPU_PARTITION','cpu'),'--time','03:55:00',
                                '--mem','128G','--cpus','64'], 'score.sbatch')
                    elif not (ROOT/'comparison/COMPLETE.json').exists() and 'interface' not in jobs:
                        jobs['interface'] = submit('interface', ['--partition',os.environ.get('CPU_PARTITION','cpu'),'--time','03:55:00',
                            '--mem','64G','--cpus','16'], 'interface.sbatch')
            if args.watch and not (ROOT/'FULL_COMPLETE.json').exists() and not list(ROOT.glob('watch*_SUBMISSION.json')):
                jobs['watch0'] = submit('watch0', ['--partition',os.environ.get('CPU_PARTITION','cpu'),'--time','03:55:00',
                    '--mem','1G','--cpus','1'], 'watch.sbatch')
        status['jobs'] = jobs
        failure = ROOT/'training/FAILED.json'
        status['training_failure'] = read(failure) if failure.exists() else None
        status['full_complete'] = all(status[k] for k in ['training_complete', 'benchmark_complete',
                                                         'interface_complete', 'comparison_complete'])
        write(ROOT/'PIPELINE_STATUS.json', status)
        if status['full_complete']:
            subprocess.run([sys.executable, str(ROOT/'finalize.py')], check=True, stdout=sys.stderr)
        print(json.dumps(status, indent=2))
    finally:
        assert read(mutex/'owner.json')==owner
        (mutex/'owner.json').unlink()
        mutex.rmdir()


if __name__ == '__main__':
    main()
