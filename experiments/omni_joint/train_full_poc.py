"""One full-diffusion Omni run; no continued-training control, no checkpoint search."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from poc_model import ROOT, ASSETS, load_model, move
from cache_training import build_entry
from full_model import FullOmni, losses, gradient_norm
from model import Observations, sample
from generate_panel import decode
from metrics import ca_lddt
from tracking import Tracker, file_hash

EXP = ROOT/'experiments/omni_joint'
RUN = EXP/'runs/full-poc-seed-1729'
LETTERS = [const.prot_token_to_letter[a] for a in const.canonical_tokens]


def rank_id(row):
    return hashlib.sha256(row['id'].encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1729)
    args = ap.parse_args()
    assert args.steps == 1000 and args.seed == 1729
    RUN.mkdir(parents=True, exist_ok=True)
    assert not (RUN/'final.pt').exists(), 'Completed run exists; do not retrain silently'
    assert not (RUN/'started_updates.json').exists(), 'Existing optimizer updates: resume explicitly, never restart silently'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    data = json.loads((EXP/'manifests/data_filtered.json').read_text())
    old_lock = json.loads((EXP/'LOCK.json').read_text())
    assert file_hash(EXP/'manifests/data_filtered.json') == old_lock['data_manifest_sha256']
    assert file_hash(ASSETS/'boltzgen1_diverse.ckpt') == old_lock['base_checkpoint_sha256']
    data_rows = {r['id']:r for split in ('train', 'validation') for r in data[split]}
    entries = json.loads((EXP/'manifests/cache.json').read_text())
    train = {m:sorted([r for r in entries if r['split']=='train' and r['mode']==m], key=rank_id)
             for m in ('folding', 'design')}
    assert all(len(v)==1613 for v in train.values())
    validation = []
    for m in ('folding', 'design'):
        validation += sorted([r for r in entries if r['split']=='validation' and r['mode']==m], key=rank_id)[:4]
    test_folding = sorted(data['test'], key=rank_id)[:20]
    motifs = [r for r in json.loads((EXP/'manifests/motifs.json').read_text()) if r['candidate']<4]
    assert len(motifs)==100 and len({r['task'] for r in motifs})==25
    atomic_json(RUN/'evaluation_manifest.json', dict(folding=test_folding, design=motifs, validation=validation,
                exploratory_previously_used_test=True, checkpoint='final.pt'))
    config = dict(steps=args.steps, seed=args.seed, effective_batch_size=1,
        task_mix={'folding':.5, 'design':.5}, learning_rate=1e-5, weight_decay=1e-4,
        warmup_steps=50, gradient_clip=1., lora=False, trunk_frozen=True,
        trainable_base_modules=['structure_module', 'diffusion_conditioning'],
        checkpoint_selection='final step 1000, shared by both tasks',
        replay_scope=['folding', 'design'], replay_input='observed identities only; hidden identities MASK',
        replay_sigma_weighted=True, replay_weight=1.,
        replay_weight_rule='same native sigma weighting and per-coordinate units as coordinate MSE; unit tradeoff, not tuned',
        identity_weight_rule='median coordinate/identity gradient norm ratio on shared denoiser, 16 design calibration batches',
        calibration_batches=32, calibration_chemical_t=.5, calibration_optimizer_updates=0,
        gpu=torch.cuda.get_device_name(), training_examples=1613, sampling_steps=500,
        base_checkpoint_sha256=old_lock['base_checkpoint_sha256'],
        data_manifest_sha256=file_hash(EXP/'manifests/data_filtered.json'),
        data_sha256=old_lock['data_sha256'], upstream_commit=old_lock['upstream_commit'],
        evaluation_manifest_sha256=file_hash(RUN/'evaluation_manifest.json'),
        protocol_sha256=file_hash(EXP/'POC_MINIMAL.md'),
        source_sha256={p.name:file_hash(p) for p in sorted(EXP.glob('*.py'))},
        auxiliary_source_sha256={p.name:file_hash(p) for p in (ROOT/'scripts/poc_model.py',ROOT/'scripts/generate_panel.py')})
    tracker = Tracker(RUN, config, 'full-diffusion-omni-poc')
    success = False
    started = time.monotonic()
    try:
        teacher = load_model()
        base = load_model()
        canonicals = load_canonicals(str(ASSETS/'mols.zip'))
        def load(row):
            if not Path(row['path']).exists():
                state=(random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
                try: build_entry(teacher, data_rows[row['id']], row['mode'], row['split'], canonicals)
                finally:
                    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])
            p = torch.load(row['path'], map_location='cpu', weights_only=False)
            assert p['hidden_coordinate_audit'] and p['id']==row['id'] and p['mode']==row['mode']
            p['cache']['native_replay_eligible'] = True
            return move(p, 'cuda')
        first=load(train['design'][0])
        model=FullOmni(base, teacher, first['cache']).cuda()
        params=[p for p in model.parameters() if p.requires_grad]
        shared=[p for p in model.base.structure_module.parameters() if p.requires_grad]
        atomic_json(RUN/'parameter_manifest.json', {n:dict(shape=list(p.shape), trainable=p.requires_grad)
                    for n,p in model.named_parameters()})
        config['trainable_parameters']=sum(p.numel() for p in params)
        # Independent frozen teacher and static trunk are mandatory invariants.
        obs=Observations(**first['observations'])
        with torch.no_grad():
            model.eval()
            noisy=first['target']+torch.randn_like(first['target'])
            sigma=noisy.new_ones(1); t=noisy.new_ones(1)
            pred,_=model.denoise(first['cache'],noisy,sigma,obs.identity,obs,t)
            native,_=model.denoise(first['cache'],noisy,sigma,obs.identity,obs,t,native=True)
            parity=float((pred-native).abs().max())
            assert parity<1e-4, ('zero-feedback parity', parity)
        model.train()
        print('AUDIT_PASSED',json.dumps(dict(zero_feedback_max_error=parity,trainable_parameters=config['trainable_parameters'],lora=False)),flush=True)
        calibration=[]
        for i in range(32):
            mode=('folding','design')[i%2]
            p=load(train[mode][i//2])
            parts,meta=losses(model,p,seed=21000+i,chemical_t_override=.5)
            norms={k:gradient_norm(v,shared) for k,v in parts.items()}
            record=dict(index=i,mode=mode,id=p['id'],losses={k:float(v.detach()) for k,v in parts.items()},gradient_norms=norms,**meta)
            assert all(np.isfinite(v) for v in norms.values())
            calibration.append(record)
            tracker.log({'calibration/'+mode+'/'+k:float(v.detach()) for k,v in parts.items()},0)
            print('CALIBRATION',i,mode,json.dumps(record['losses']),flush=True)
            del parts,p
        ratios=[r['gradient_norms']['coordinate']/r['gradient_norms']['identity']
                for r in calibration if r['mode']=='design' and r['gradient_norms']['identity']>0]
        assert len(ratios)==16
        identity_weight=float(np.median(ratios))
        assert np.isfinite(identity_weight) and identity_weight>0
        config['identity_weight']=identity_weight
        config['calibration_note']='Replay gradient is zero at identical initialization; its unit weight is a declared tradeoff after sigma/unit normalization, not a fitted gradient ratio.'
        atomic_json(RUN/'calibration.json',dict(batches=calibration,identity_weight=identity_weight,replay_weight=1.,note=config['calibration_note']))
        atomic_json(RUN/'config.json',config)
        tracker.run.config.update(config,allow_val_change=True)
        print('WEIGHTS_FIXED',identity_weight,1.,flush=True)
        # The calibration does not update any model parameters or optimizer state.
        optimizer=torch.optim.AdamW(params,lr=config['learning_rate'],weight_decay=config['weight_decay'])
        del first
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
        orders={m:[] for m in train}
        snapshots=[]
        def validate(step):
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            rows=[]
            try:
                model.eval()
                for i,row in enumerate(validation):
                    p=load(row);obs=Observations(**p['observations']);f=p['cache']['conditioning']['feats']
                    with torch.no_grad():
                        parts,_=losses(model,p,seed=31000+i)
                        result=sample(model,p['cache'],obs,steps=500,seed=41000+i)
                    x=result['coordinates'];seq,_,_=decode(x,f)
                    ids=torch.tensor([[LETTERS.index(a) if a in LETTERS else -1 for a in seq]],device=x.device)
                    editable=~obs.identity_mask
                    n=p['labels'].shape[1]
                    bbmap=f['token_to_bb4_atoms'].float().reshape(1,n*4,-1)
                    bb=torch.bmm(bbmap,x)[0].reshape(n,4,3).cpu().numpy()
                    true=torch.bmm(bbmap,p['target'])[0].reshape(n,4,3).cpu().numpy()
                    distances=np.linalg.norm(np.diff(bb[:,1],axis=0),axis=-1)
                    valid=bool(torch.all((ids[editable]>=0)&(ids[editable]==result['identity'][editable])))
                    item=dict(step=step,id=row['id'],mode=row['mode'],
                        ca_lddt=ca_lddt(bb[:,1],true[:,1]),geometry_unknown=int((ids<0).sum()),
                        abnormal_ca_distances=int(((distances<3.4)|(distances>4.2)).sum()),
                        identity_geometry_agreement=float((ids[editable]==result['identity'][editable]).float().mean()) if editable.any() else 1.,
                        endpoint_valid=valid and bool(np.isfinite(bb).all()) and bool((ids>=0).all()),
                        losses={k:float(v) for k,v in parts.items()})
                    rows.append(item)
                    print('ROLLOUT_VALIDATION',json.dumps(item),flush=True)
                atomic_json(RUN/f'validation-{step}.json',rows)
                for mode in ('folding','design'):
                    chosen=[r for r in rows if r['mode']==mode]
                    tracker.log({'validation/'+mode+'/'+key:float(np.mean([r[key] for r in chosen]))
                        for key in ('ca_lddt','geometry_unknown','abnormal_ca_distances','identity_geometry_agreement','endpoint_valid')},step)
                    tracker.log({'validation/'+mode+'/loss_'+key:float(np.mean([r['losses'][key] for r in chosen]))
                        for key in ('coordinate','identity','replay')},step)
                return rows
            finally:
                random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3]);model.train()
        for step in range(1,args.steps+1):
            mode=('folding','design')[(step-1)%2]
            if not orders[mode]: orders[mode]=random.sample(train[mode],len(train[mode]))
            row=orders[mode].pop();p=load(row)
            optimizer.zero_grad(set_to_none=True)
            parts,meta=losses(model,p)
            value=parts['coordinate']+identity_weight*parts['identity']+parts['replay']
            assert torch.isfinite(value), 'Nonfinite objective'
            component_norms={}
            if step<=2 or step%50 in (0,1):
                component_norms={k:gradient_norm(v,shared) for k,v in parts.items()}
            value.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            lr=config['learning_rate']*min(1.,step/config['warmup_steps'])
            for group in optimizer.param_groups: group['lr']=lr
            if step==1: atomic_json(RUN/'started_updates.json',dict(first_step=1))
            optimizer.step()
            record={'train/loss':float(value.detach()),'train/'+mode+'/loss':float(value.detach()),
                **{'train/'+mode+'/'+k:float(v.detach()) for k,v in parts.items()},
                **{'train/'+mode+'/gradient_'+k:v for k,v in component_norms.items()},
                'train/weighted_identity':identity_weight*float(parts['identity'].detach()),
                'train/weighted_replay':float(parts['replay'].detach()),
                'train/replay_unweighted':meta['replay_unweighted'],'train/gradient_norm_before_clip':grad,
                'train/learning_rate':lr,'compute/GPU_hours':(time.monotonic()-started)/3600,
                'compute/peak_memory_GB':torch.cuda.max_memory_allocated()/1e9}
            tracker.log(record,step)
            if step<=2 or step%10==0: print('TRAIN_FULL',step,mode,json.dumps(record),flush=True)
            del parts,p,value
            if step%250==0:
                scores=validate(step)
                state=dict(adapter=model.trainable_state(),step=step,seed=args.seed,config=config,validation=scores,
                           architecture='full-diffusion-no-lora',shared_across_tasks=True)
                torch.save(state,RUN/'last.pt')
                if step==args.steps:
                    (RUN/'last.pt').replace(RUN/'final.pt')
                    tracker.checkpoint(RUN/'final.pt',selected=True)
                    passed=any(r['endpoint_valid'] for r in scores if r['mode']=='design')
                    atomic_json(RUN/'evaluation_gate.json',dict(run_completed=True,full_evaluation_allowed=passed,
                        reason='At least one valid validation design' if passed else 'All four final validation designs failed raw endpoint validity',
                        joint_success_established=False))
                del state
        atomic_json(RUN/'completion.json',dict(training_completed=True,steps=args.steps,seconds=time.monotonic()-started))
        success=True
    except Exception as e:
        # Do not serialize exception bodies that could include environment details.
        atomic_json(RUN/'failure.json',dict(exception_type=type(e).__name__,training_completed=False))
        raise
    finally:
        tracker.finish(success)


if __name__=='__main__': main()
