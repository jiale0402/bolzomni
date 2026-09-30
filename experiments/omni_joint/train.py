"""One joint checkpoint; refuses to train before protocol/architecture/data gates."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
import numpy as np
from cache_training import build_entry
from boltzgen.data.mol import load_canonicals
from poc_model import ASSETS
from poc_model import ROOT, load_model, move
from model import OmniCompletion, Observations, completion_loss
from tracking import Tracker, file_hash

EXP=ROOT/'experiments/omni_joint'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=6000)
    ap.add_argument('--seed',type=int,default=1729); args=ap.parse_args()
    lock=json.loads((EXP/'LOCK.json').read_text())
    required=('native_reproduction','architecture_audit','compiler_leakage_audit',
              'dataset_overlap_audit','folding_baseline_qualified','design_baseline_qualified',
              'evaluator_qualified','cost_profile_approved')
    assert all(lock.get(k) is True for k in required),'Experiment qualification incomplete'
    assert lock['protocol_sha256']==file_hash(EXP/'PROTOCOL.md')
    assert lock['cache_manifest_sha256']==file_hash(EXP/'manifests/cache.json')
    for relative,expected in lock['source_sha256'].items():
        assert file_hash(EXP/relative)==expected,('Source changed after locking',relative)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed); random.seed(args.seed)
    entries=json.loads((EXP/'manifests/cache.json').read_text())
    train=[r for r in entries if r['split']=='train']
    validation=[]
    for mode in ('folding','design','inverse','mixed'):
        validation.extend([r for r in entries if r['split']=='validation' and r['mode']==mode][:10])
    assert {r['mode'] for r in train}=={'folding','design','inverse','mixed'}
    base=load_model();canonicals=load_canonicals(str(ASSETS/'mols.zip'))
    data=json.loads((EXP/'manifests/data_filtered.json').read_text())
    data_rows={r['id']:r for split in ('train','validation') for r in data[split]}
    def load(row):
        if not Path(row['path']).exists():
            states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            try:build_entry(base,data_rows[row['id']],row['mode'],row['split'],canonicals)
            finally:
                random.setstate(states[0]);np.random.set_state(states[1])
                torch.set_rng_state(states[2]);torch.cuda.set_rng_state_all(states[3])
        payload=torch.load(row['path'],map_location='cpu',weights_only=False)
        assert payload['hidden_coordinate_audit'] and payload['id']==row['id'] and payload['mode']==row['mode']
        return move(payload,'cuda')
    first=load(train[0]); cache=first['cache']
    model=OmniCompletion(base,cache['conditioning']['s_trunk'].shape[-1],cache['z_trunk'].shape[-1],
                         base.structure_module.score_model.a_norm.normalized_shape[0]).cuda()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
    run_dir=EXP/'runs'/f'joint-seed-{args.seed}'
    run_dir.mkdir(parents=True,exist_ok=True)
    tracker=Tracker(run_dir,dict(lock=lock,steps=args.steps,seed=args.seed,learning_rate=1e-4,
            identity_weight=0.1,replay_weight=1.,rank=8,trainable_parameters=sum(p.numel() for p in parameters),
            training_examples=len(train),validation_prompts=len(validation),gpu=torch.cuda.get_device_name(),
            checkpoint_selection='mean four-mode validation loss / initial mode loss'),'joint-shared-checkpoint')
    def loss_for(payload,seed=None):
        return completion_loss(model,payload['cache'],payload['target'],payload['labels'],payload['resolved'],
                               Observations(**payload['observations']),seed=seed)
    def validate(step=0):
        model.eval(); totals={m:[] for m in ('folding','design','inverse','mixed')}
        components={}
        with torch.no_grad():
            for i,row in enumerate(validation):
                value,parts=loss_for(load(row),10000+i)
                totals[row['mode']].append(float(value))
                for key,value_part in parts.items():
                    components.setdefault('validation/'+row['mode']+'_'+key,[]).append(float(value_part))
        model.train()
        tracker.log({key:sum(values)/len(values) for key,values in components.items()},step)
        return {m:sum(v)/len(v) for m,v in totals.items()}
    success=False
    try:
        initial=validate(); best=float('inf'); order=[]; started=time.monotonic()
        for step in range(1,args.steps+1):
            if not order: order=random.sample(train,len(train))
            row=order.pop(); payload=load(row)
            optimizer.zero_grad(set_to_none=True)
            value,parts=loss_for(payload)
            if not torch.isfinite(value): raise RuntimeError('Nonfinite objective')
            value.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step()
            if step%10==0:
                tracker.log({'train/loss':float(value.detach()),
                    **{'train/'+k:float(v) for k,v in parts.items()},
                    'compute/peak_memory_GB':torch.cuda.max_memory_allocated()/1e9,
                    'compute/training_GPU_hours':(time.monotonic()-started)/3600},step)
                print('TRAIN',step,float(value.detach()),flush=True)
            if step%500==0 or step==args.steps:
                scores=validate(step)
                joint=sum(scores[m]/max(initial[m],1e-8) for m in scores)/len(scores)
                tracker.log({**{'validation/'+m+'_loss':v for m,v in scores.items()},'validation/joint_selection':joint},step)
                selected=joint<best
                state=dict(adapter=model.trainable_state(),optimizer=optimizer.state_dict(),step=step,
                           validation=scores,joint_selection=joint,config=lock,seed=args.seed)
                torch.save(state,run_dir/'last.pt')
                if selected:
                    best=joint; torch.save(state,run_dir/'selected.pt')
                    tracker.checkpoint(run_dir/'selected.pt',selected=True)
                torch.manual_seed(args.seed+step)
        success=True
    finally:
        tracker.finish(success)


if __name__=='__main__':main()
