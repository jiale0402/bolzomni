"""Mandatory W&B tracking with local scalar journal; no environment/config secret dumps."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
import wandb


def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


class Tracker:
    def __init__(self, directory, config, name, job_type='train'):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        # Caller supplies a deliberate experiment config, never os.environ or argparse
        # dumps containing tokens. Reject accidentally introduced credential keys.
        def check(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if any(s in str(key).lower() for s in ('password','api_key','secret','access_token')):
                        raise ValueError('Credential-like field is forbidden in tracking config')
                    check(item)
            elif isinstance(value,(list,tuple)):
                for item in value: check(item)
        check(config)
        id_path = self.directory/'wandb_run_id.txt'
        run_id = id_path.read_text().strip() if id_path.exists() else uuid.uuid4().hex[:12]
        id_path.write_text(run_id)
        requested_mode = os.environ.get('WANDB_MODE','online')
        args = dict(project='boltz-omni-joint',id=run_id,name=name,job_type=job_type,
                    config=config,dir=str(self.directory),resume='allow',
                    settings=wandb.Settings(init_timeout=45,disable_git=True))
        if os.environ.get('WANDB_ENTITY'): args['entity'] = os.environ['WANDB_ENTITY']
        self.mode = requested_mode
        try:
            self.run = wandb.init(mode=requested_mode,**args)
        except (wandb.errors.Error,TimeoutError) as error:
            # Avoid printing service response bodies, which can contain credential details.
            print('WANDB_ONLINE_UNAVAILABLE',type(error).__name__,'OFFLINE_PENDING_SYNC',flush=True)
            self.mode = 'offline'
            self.run = wandb.init(mode='offline',**args)
        self.logfile = open(self.directory/'metrics.jsonl','a',buffering=1)
        self.started = time.monotonic()
        self.run.define_metric('optimizer_step')
        self.run.define_metric('train/*',step_metric='optimizer_step')
        self.run.define_metric('validation/*',step_metric='optimizer_step')
        state = dict(id=run_id,mode=self.mode,url=self.run.url if self.mode=='online' else None,
                     sync_required=self.mode=='offline')
        (self.directory/'tracking_status.json').write_text(json.dumps(state,indent=2))
        print('WANDB_TRACKING',json.dumps(state),flush=True)

    def log(self, values, optimizer_step):
        record = dict(values,optimizer_step=int(optimizer_step),wall_seconds=time.monotonic()-self.started)
        self.logfile.write(json.dumps(record)+'\n')
        self.run.log(record)

    def checkpoint(self, path, selected=False):
        artifact = wandb.Artifact(f'checkpoint-{self.run.id}',type='model',
                                 metadata={'sha256':file_hash(path),'shared_across_tasks':True})
        artifact.add_file(str(path))
        self.run.log_artifact(artifact,aliases=['selected' if selected else 'latest'])

    def finish(self, success=True):
        self.logfile.close()
        self.run.finish(exit_code=0 if success else 1)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--directory',required=True); args=ap.parse_args()
    tracker=Tracker(args.directory,{'purpose':'tracking-connectivity-check','training_started':False},
                    'tracking-connectivity-check',job_type='diagnostic')
    tracker.log({'diagnostic/tracking_check':1},0)
    tracker.finish()
