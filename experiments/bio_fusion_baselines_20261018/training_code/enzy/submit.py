"""Submit a configured Slurm job; account and site settings are external."""
import argparse,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 for name in ['partition','time','mem','cpus','gres','constraint','array','dependency','name']:
  p.add_argument('--'+name,required=name in ['partition','time','name'])
 p.add_argument('--account',default=os.getenv('SLURM_ACCOUNT'))
 p.add_argument('script');p.add_argument('script_args',nargs='*');a=p.parse_args()
 if not a.account:p.error('Set SLURM_ACCOUNT or provide --account')
 root=Path(os.environ['FUSION_ROOT']);(root/'logs').mkdir(parents=True,exist_ok=True)
 cmd=['sbatch','--parsable','--account='+a.account,'--partition='+a.partition,'--time='+a.time,
      '--job-name='+a.name,'--chdir='+str(root),'--output='+str(root/'logs'/(a.name+'-%A_%a.log')),
      '--export=ALL,FUSION_ROOT='+str(root)]
 for name,flag in [('mem','mem'),('cpus','cpus-per-task'),('gres','gres'),('constraint','constraint'),('array','array'),('dependency','dependency')]:
  if getattr(a,name):cmd.append('--'+flag+'='+getattr(a,name))
 env={k:v for k,v in os.environ.items() if k not in ['PYTHONPATH','PYTHONHOME']}
 test=subprocess.run(cmd+['--test-only',a.script]+a.script_args,capture_output=True,text=True,env=env)
 if test.returncode:raise RuntimeError(test.stderr)
 result=subprocess.run(cmd+[a.script]+a.script_args,capture_output=True,text=True,env=env)
 audit=dict(utc=datetime.now(timezone.utc).isoformat(),account=a.account,request=vars(a),
            stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
 with (root/'submissions.jsonl').open('a') as f:f.write(json.dumps(audit)+'\n')
 print(json.dumps(audit,indent=2))
 if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
