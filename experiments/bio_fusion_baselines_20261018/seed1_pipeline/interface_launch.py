"""Prepare an isolated post-hoc scorer without changing committed benchmark data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    run = json.loads((ROOT / 'benchmark/run.json').read_text())
    train = json.loads((ROOT / 'code/config.json').read_text())
    dest = ROOT / 'interface_metrics'
    dest.mkdir(exist_ok=True)
    code = dest / 'code'
    code.mkdir(exist_ok=True)
    config = dict(run_name=train['run_name'], training_seed=train['seed'], checkpoint_sha256=run['checkpoint_sha256'],
                  benchmark_root=str(ROOT / 'benchmark'), benchmark_run_hash=run['run_hash'],
                  anew_source=run['runtime']['anew_source'], wandb_entity=train['wandb_entity'], wandb_project=train['wandb_project'])
    protocol = dict(operator_source='Unmodified main-table evaluate.py (DockQ 2.1.3), score_more.py backbone lDDT, native_metrics.py and residue_mapping.py',
                    scope={'lnr': 'peptide vs target', 'rabd': 'HCDR3 vs antigen only; excludes antibody framework'},
                    aggregation='Finite candidate values within target, then equal target means; retain missing denominators',
                    inputs='Existing 153 protein targets x 30 fixed slots; no new generation; no checkpoint selection',
                    checkpoint_sha256=run['checkpoint_sha256'], benchmark_run_hash=run['run_hash'],
                    benchmark_previously_inspected=True, known_ligand_training_pdb_overlap=19)
    sources = ROOT / 'interface_sources'
    payload = {p.name: p.read_bytes() for p in sources.iterdir() if p.is_file()}
    payload['config.json'] = (json.dumps(config, indent=2)+'\n').encode()
    payload['protocol.json'] = (json.dumps(protocol, indent=2)+'\n').encode()
    for name, data in payload.items():
        p = code / name
        if p.exists():
            assert p.read_bytes() == data, 'Never overwrite started scorer inputs: '+name
        else:
            p.write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    ledger = dest / 'DEPLOYED.json'
    if ledger.exists():
        assert json.loads(ledger.read_text()) == hashes
    else:
        ledger.write_text(json.dumps(hashes, indent=2)+'\n')
    complete = dest / 'COMPLETE.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        assert record['checkpoint_sha256'] == config['checkpoint_sha256'] and record['benchmark_run_hash'] == run['run_hash']
        assert hashlib.sha256((dest / 'candidates.jsonl').read_bytes()).hexdigest() == record['candidate_sha256']
        return
    env = dict(os.environ, FUSION_ROOT=str(dest), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    subprocess.run([run['runtime']['score_python'], '-u', str(code / 'interface_score.py')], env=env, cwd=dest, check=True)


if __name__ == '__main__':
    main()
