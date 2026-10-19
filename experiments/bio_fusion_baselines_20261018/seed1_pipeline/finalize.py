"""Hash verified completion; retain post-run accounting separately from scores."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths = ['code/config.json', 'SOURCES.json', 'UPSTREAM_SOURCES.json', 'training/COMPLETE.json',
        'training/provenance.json', 'training/selection.json', 'profile/NATIVE_PREFLIGHT.json',
        'benchmark/COMPLETE.json', 'benchmark/REPORT.json', 'benchmark/run.json',
        'interface_metrics/COMPLETE.json', 'interface_metrics/SUMMARY.json', 'comparison/RESULTS.json']
    for name in paths:
        assert (ROOT/name).exists(), name
    cfg = json.loads((ROOT/'code/config.json').read_text())
    train = json.loads((ROOT/'training/COMPLETE.json').read_text())
    bench = json.loads((ROOT/'benchmark/run.json').read_text())
    assert train['base_unchanged'] and train['frozen_parameters_unchanged'] and not train['budget_exceeded']
    assert sha(ROOT/'training/best_joint.pt')==train['checkpoint_sha256']==bench['checkpoint_sha256']
    assert json.loads((ROOT/'profile/NATIVE_PREFLIGHT.json').read_text())['passed']
    subprocess.run(['python3',str(ROOT/'delivery.py')],check=True)
    tracking = []
    for p in list((ROOT/'tracking').glob('*.status.json')) + list((ROOT/'benchmark/tracking').glob('*.status.json')) + list((ROOT/'interface_metrics/tracking').glob('*.status.json')):
        tracking.append(json.loads(p.read_text()))
    assert tracking, 'Missing W&B/offline receipts'
    output = ROOT/'FULL_COMPLETE.json'
    report = dict(completed=True, method=cfg['method_id'], training_seed=cfg['seed'],
        budget_seconds=cfg['budget_seconds'], targets=253, attempted_slots=7590,
        input_manifest_sha256=cfg['manifest_sha256'], checkpoint_sha256=train['checkpoint_sha256'],
        receipts_sha256={name: sha(ROOT/name) for name in paths}, tracking=tracking,
        pending_offline_sync=any(r.get('pending_sync', True) for r in tracking),
        wandb_server_verified=False, server_verification_note='Local lifecycle receipts only; verify finished/COMMITTED remotely before final delivery',
        benchmark_previously_inspected=True, known_ligand_training_pdb_overlap=19)
    if output.exists():
        assert json.loads(output.read_text())==report
    else:
        output.write_text(json.dumps(report, indent=2)+'\n')
    jobs = [json.loads(p.read_text())['stdout'].strip() for p in ROOT.glob('*_SUBMISSION.json')]
    result = subprocess.run(['sacct','-n','-P','-j',','.join(jobs),'-o',
        'JobID,JobName,Account,State,ElapsedRaw,AllocCPUS,AllocTRES,NodeList'], capture_output=True, text=True)
    (ROOT/'POST_RUN_ACCOUNTING.txt').write_text(result.stdout)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
