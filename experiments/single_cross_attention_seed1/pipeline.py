"""Idempotent Slurm pipeline; every submission refreshes FairShare/eligibility."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def submit(stage,args,script):
    path=ROOT/(stage+'_SUBMISSION.json')
    assert not path.exists()
    result=subprocess.run(['python3',str(ROOT/'code/submit.py'),*args,'--name','single-ca1-'+stage,
        str(ROOT/script)],env=dict(os.environ,FUSION_ROOT=str(ROOT)),capture_output=True,text=True)
    if result.returncode:
        write(ROOT/(stage+'_SUBMIT_FAILED.json'),{'stderr':result.stderr,'stdout':result.stdout,'unix':time.time()})
        raise RuntimeError(result.stdout+result.stderr)
    receipt=json.loads(result.stdout)
    write(path,receipt)
    return receipt['stdout'].strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--advance',action='store_true');a=p.parse_args()
    with (ROOT/'PIPELINE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        report={'unix':time.time(),'root':str(ROOT),'training_complete':(ROOT/'training/COMPLETE.json').exists(),
            'benchmark_complete':(ROOT/'benchmark/COMPLETE.json').exists(),
            'comparison_complete':(ROOT/'comparison/COMPLETE.json').exists()}
        jobs={}
        for stage in ['train','generation','scoring']:
            path=ROOT/(stage+'_SUBMISSION.json')
            if path.exists():jobs[stage]=read(path)['stdout'].strip()
        report['jobs']=jobs
        if jobs:
            result=subprocess.run(['sacct','-n','-P','-j',','.join(jobs.values()),
                '-o','JobID,State,Elapsed,AllocTRES'],capture_output=True,text=True)
            report['slurm']=result.stdout
        if a.advance:
            if not report['training_complete']:
                if 'train' not in jobs:
                    jobs['train']=submit('train',['--partition','gengpu','--time','04:15:00',
                        '--mem','64G','--cpus','4','--gres','gpu:a100:1'],'train.sbatch')
            else:
                subprocess.run([sys.executable,str(ROOT/'prepare_benchmark.py')],check=True)
                if 'generation' not in jobs:
                    jobs['generation']=submit('generation',['--partition','gengpu','--time','04:00:00',
                        '--mem','64G','--cpus','4','--gres','gpu:a100:1','--array','0-3%4'],'generate.sbatch')
                chunks=list((ROOT/'benchmark/generation/omni').glob('COMPLETE-chunk*.json'))
                report['generation_chunks_complete']=len(chunks)
                if len(chunks)==31:
                    records=[read(p) for p in sorted(chunks)]
                    cfg=read(ROOT/'benchmark/run.json')
                    assert all(r['provenance']['run_hash']==cfg['run_hash'] and r['attempted']==r['expected'] for r in records)
                    assert sum(r['attempted'] for r in records)==7590
                    write(ROOT/'benchmark/generation/omni/COMPLETE-shard0.json',
                        {'provenance':records[0]['provenance'],'attempted':7590,'expected':7590,
                         'successful':sum(r['successful'] for r in records),'targets':253,'chunks':31,
                         'fixed_attempts_per_target':30,'generation_complete':True})
                    if 'scoring' not in jobs:
                        jobs['scoring']=submit('scoring',['--partition','short','--time','03:55:00',
                            '--mem','128G','--cpus','64'],'score.sbatch')
        report['jobs']=jobs
        report['training_failure']=read(ROOT/'training/FAILED.json') if (ROOT/'training/FAILED.json').exists() else None
        write(ROOT/'PIPELINE_STATUS.json',report)
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
