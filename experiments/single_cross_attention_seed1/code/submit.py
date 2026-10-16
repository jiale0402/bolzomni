"""Submit only after a fresh user FairShare comparison and resource eligibility test."""
import argparse
import datetime
import json
import math
import os
import subprocess
from pathlib import Path

ACCOUNTS = ('p53127', 'p53252')

def command(args):
    # Slurm epilogues can submit the next stage from a different Python ABI.
    # Keep stage-specific module paths out of the next sbatch --export=ALL.
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    return subprocess.run(args, text=True, capture_output=True, env=env)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--partition', choices=['short', 'gengpu'], required=True)
    p.add_argument('--time', required=True)
    p.add_argument('--mem', default='32G')
    p.add_argument('--cpus', default='4')
    p.add_argument('--gres')
    p.add_argument('--array')
    p.add_argument('--dependency')
    p.add_argument('--name', required=True)
    p.add_argument('script')
    p.add_argument('script_args', nargs='*')
    a = p.parse_args()
    root = Path(os.environ.get('FUSION_ROOT', '/projects/p53127/anewomni_work/pairformer_native10000_20260929'))
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    output = command(['sshare', '-n', '-P', '-l', '-u', 'jck3114', '-A', ','.join(ACCOUNTS),
                      '--format=Account,User,FairShare'])
    if output.returncode:
        raise RuntimeError(output.stderr)
    shares = {}
    for line in output.stdout.splitlines():
        fields = [x.strip() for x in line.split('|')]
        if len(fields) >= 3 and fields[0] in ACCOUNTS and fields[1] == 'jck3114':
            value = float(fields[2])
            assert math.isfinite(value)
            shares[fields[0]] = value
    assert set(shares) == set(ACCOUNTS), f'Both user FairShares required: {shares}'
    assoc = command(['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', 'user=jck3114',
                     'account=' + ','.join(ACCOUNTS), 'format=Account,User,Partition,QOS,DefaultQOS'])
    assert assoc.returncode == 0, assoc.stderr
    base = ['sbatch', '--parsable', '--partition=' + a.partition, '--time=' + a.time,
            '--mem=' + a.mem, '--cpus-per-task=' + a.cpus, '--job-name=' + a.name,
            '--output=' + str(root / 'logs' / (a.name + '-%A_%a.log')),
            '--chdir=' + str(root), '--export=ALL,FUSION_ROOT=' + str(root)]
    if a.gres:
        base.append('--gres=' + a.gres)
    if a.array:
        base.append('--array=' + a.array)
    if a.dependency:
        base.append('--dependency=' + a.dependency)
    tests = {}
    eligible = []
    for account in ACCOUNTS:
        rows = [r.split('|') for r in assoc.stdout.splitlines()]
        associated = any(len(r) >= 4 and r[0].strip() == account and r[1].strip() == 'jck3114'
                         and r[2].strip() in ('', a.partition) and 'normal' in r[3] for r in rows)
        result = command(base + ['--account=' + account, '--test-only', a.script] + a.script_args)
        tests[account] = {'association_ok': associated, 'returncode': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr}
        if associated and result.returncode == 0:
            eligible.append(account)
    assert eligible, tests
    account = max(eligible, key=lambda x: (shares[x], x))
    result = command(base + ['--account=' + account, a.script] + a.script_args)
    audit = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'user': 'jck3114', 'user_fairshare': shares, 'associations': assoc.stdout,
             'eligibility_tests': tests, 'chosen_account': account, 'request': vars(a),
             'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode,
             'environment_isolation': 'PYTHONPATH and PYTHONHOME removed before sbatch; each stage selects its own interpreter'}
    audit_path = root / 'submissions.jsonl'
    with audit_path.open('a') as f:
        f.write(json.dumps(audit) + '\n')
    print(json.dumps(audit, indent=2), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
