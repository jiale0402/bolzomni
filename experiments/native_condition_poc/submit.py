"""Pick the currently eligible higher-user-FairShare Quest account for one job."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('script');ap.add_argument('--dependency');args=ap.parse_args()
    raw=subprocess.check_output(['sshare','-nP','-A','p53127,p53252','-u','jck3114','-o','Account,User,FairShare'],text=True)
    scores={}
    for line in raw.splitlines():
        p=line.split('|')
        if len(p)>=3 and p[1].strip()=='jck3114' and p[0].strip() in ('p53127','p53252'):
            scores[p[0].strip()]=float(p[2])
    assert set(scores)=={'p53127','p53252'},'Missing current user-level FairShare'
    tests={};eligible=[]
    for account in sorted(scores):
        command=['sbatch','--test-only','--account='+account]
        if args.dependency:command+=['--dependency='+args.dependency]
        response=subprocess.run(command+[args.script],capture_output=True,text=True)
        tests[account]={'eligible':response.returncode==0,'output':response.stdout+response.stderr}
        if response.returncode==0:eligible.append(account)
    assert eligible, 'Neither allocation accepts requested resources'
    account=max(eligible,key=lambda a:scores[a])
    command=['sbatch','--parsable','--account='+account]
    if args.dependency:command+=['--dependency='+args.dependency]
    job_id=subprocess.check_output(command+[args.script],text=True).strip()
    record=dict(submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script=args.script,
                user_fairshare=scores,eligibility=tests,account=account,job_id=job_id,dependency=args.dependency)
    dest=Path(__file__).parent/'submissions';dest.mkdir(exist_ok=True)
    (dest/(job_id+'.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
