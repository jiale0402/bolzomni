"""One shared, corrected full-diffusion Omni on verified official training members."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from boltzgen.data.mol import load_canonicals
from poc_model import ROOT as PROJECT, ASSETS, load_model, move
import cache_training
from model import Observations
from native_flow import native_observations, FullOmni, losses, gradient_norm, augmented_target, refresh_teacher_bridge, sample
from generate_panel import decode
from metrics import ca_lddt
from tracking import Tracker, file_hash

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent/'official_poc'
RUN = ROOT/'runs/native-1729'


def save_json(path, obj):
    path = Path(path);tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2));tmp.replace(path)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    resume_path = RUN/'resume.pt'
    assert not (RUN/'started_updates.json').exists() or resume_path.exists(), 'Updated run requires saved state'
    assert not (RUN/'completion.json').exists(), 'Completed run must not be trained again'
    resume = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path.exists() else None
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    random.seed(1729);np.random.seed(1729);torch.manual_seed(1729)
    data = json.loads((DATA/'data_manifest.json').read_text())
    assert len(data['train']) == 128 and len(data['validation']) == 16
    assert all(r['source']=='boltzgen/boltzgen1_train' for r in data['train'])
    cfg = dict(seed=1729, steps=400, task_mix={'folding':.5,'design':.5}, gradient_accumulation=8,
        effective_batch_size=8, frozen_trunk=True, lora=False, diffusion_lr=5e-6, new_module_lr=1e-4,
        warmup=40, gradient_clip=10., optimizer='AdamW', betas=[.9,.95], weight_decay=1e-4,
        coordinate_augmentation=True, native_smooth_lddt=True, replay='sigma-weighted MSE, both tasks',
        replay_weight=1., identity_weight_rule='minimum of median geometry/CE gradient ratios over ALL parameters and shared diffusion; 16 fixed calibration batches',
        calibration_sigma=[.5,2.,8.,32.], observed_atoms='no absolute observations; native relative structure groups retained',
        checkpoint_selection='fixed final update 400; one shared checkpoint for both tasks',
        data_manifest_sha256=file_hash(DATA/'data_manifest.json'), base_checkpoint_sha256=file_hash(ASSETS/'boltzgen1_diverse.ckpt'),
        sources={str(p.relative_to(PROJECT)):file_hash(p) for p in [*ROOT.glob('*.py'),PROJECT/'experiments/omni_joint/model.py',PROJECT/'experiments/omni_joint/full_model.py',PROJECT/'experiments/omni_joint/cache_training.py',DATA/'corrected_model.py',PROJECT/'upstream/src/boltzgen/model/modules/diffusion.py']},
        comparison='untouched original checkpoint only; no control training or extra training ablations',
        GPU=torch.cuda.get_device_name())
    tracker = Tracker(RUN,cfg,'native-condition-joint-omni')
    start=time.monotonic();ok=False
    try:
        teacher=load_model();base=load_model();canonicals=load_canonicals(str(ASSETS/'mols.zip'))
        cache_training.EXP=DATA
        def load(row, mode, split):
            path=DATA/'cache'/f'{row["id"]}-{mode}.pt'
            if not path.exists():
                state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
                try:cache_training.build_entry(teacher,row,mode,split,canonicals,observed_backbone_only=True)
                finally:
                    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])
            p=move(torch.load(path,map_location='cpu',weights_only=False),'cuda')
            assert p['hidden_coordinate_audit']
            hard=p['observations']['coordinate_mask']
            assert not (hard & ~p['cache']['conditioning']['feats']['backbone_mask'].bool()).any()
            refresh_teacher_bridge(teacher,p['cache'])
            return native_observations(p)
        first=load(data['train'][0],'design','train')
        model=FullOmni(base,teacher,first['cache']).cuda()
        parameters=[p for p in model.parameters() if p.requires_grad]
        shared=[p for p in model.base.structure_module.parameters() if p.requires_grad]
        cfg['trainable_parameters']=sum(p.numel() for p in parameters)
        # Actual initialization is archived; no exact original-output equality gate.
        completed_step = 0
        if resume:
            assert resume['architecture'] == 'native-condition-full-diffusion-no-lora'
            model.load_state_dict(resume['adapter'], strict=False)
        else:
            torch.save(dict(adapter=model.trainable_state(), step=0, config=cfg,
                architecture='native-condition-full-diffusion-no-lora'), RUN/'initial.pt')
            tracker.checkpoint(RUN/'initial.pt')
        model.eval()
        with torch.no_grad():
            target, newobs = augmented_target(first)
            mask=first['resolved'][0].bool(); before=first['target'][0,mask][::13]; after=target[0,mask][::13]
            distance_error=float((torch.cdist(before.double(),before.double(),compute_mode='donot_use_mm_for_euclid_dist')-
                torch.cdist(after.double(),after.double(),compute_mode='donot_use_mm_for_euclid_dist')).abs().max())
            assert distance_error < 1e-3 and not newobs.coordinate_mask.any()
            seq,_,_=decode(first['target'],first['cache']['conditioning']['feats'])
            assert seq == data['train'][0]['sequence']
        save_json(RUN/'preflight.json',dict(augmentation_distance_error=distance_error,
            exact_output_reproduction_required=False, native_relative_conditions=True,
            full_sampler_check='validation-0.json before any update', official_target_roundtrip=True))
        model.train()
        calibration=json.loads((RUN/'calibration.json').read_text())['batches'] if resume else []
        for i in range(0 if resume else 16):
            p=load(data['train'][i],'design','train')
            parts,meta=losses(model,p,seed=22000+i,chemical_t_override=.5,sigma_override=cfg['calibration_sigma'][i%4])
            geometry=parts['coordinate']+parts['smooth_lddt']
            row=dict(index=i,losses={k:float(v.detach()) for k,v in parts.items()},**meta)
            row['all_geometry']=gradient_norm(geometry,parameters);row['all_identity']=gradient_norm(parts['identity'],parameters)
            row['shared_geometry']=gradient_norm(geometry,shared);row['shared_identity']=gradient_norm(parts['identity'],shared)
            assert all(np.isfinite(row[k]) for k in ('all_geometry','all_identity','shared_geometry','shared_identity'))
            calibration.append(row)
            print('CALIBRATION',i,json.dumps(row),flush=True)
            del p,parts,geometry
        weight=min(float(np.median([r[k+'_geometry']/max(r[k+'_identity'],1e-12) for r in calibration])) for k in ('all','shared'))
        assert np.isfinite(weight) and weight>0
        cfg['identity_weight']=weight
        save_json(RUN/'calibration.json',dict(batches=calibration,identity_weight=weight))
        save_json(RUN/'config.json',cfg);tracker.run.config.update(cfg,allow_val_change=True)
        tracker.log({'calibration/identity_weight':weight,'preflight/augmentation_distance_error':distance_error},0)
        native_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')]
        new_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('adapter.')]
        optimizer=torch.optim.AdamW([{'params':native_params,'lr':cfg['diffusion_lr'],'peak_lr':cfg['diffusion_lr']},
            {'params':new_params,'lr':cfg['new_module_lr'],'peak_lr':cfg['new_module_lr']}],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
        def validate(step, rollout=False):
            state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            model.eval();rows=[]
            try:
                for split in ('train','validation'):
                    for i,row in enumerate(data[split][:4 if split=='train' else 16]):
                        for mode in ('folding','design'):
                            p=load(row,mode,split)
                            with torch.no_grad():parts,meta=losses(model,p,seed=31000+i,chemical_t_override=.5,sigma_override=8.)
                            rec=dict(split=split,id=row['id'],mode=mode,**{k:float(v) for k,v in parts.items()})
                            if rollout and split=='validation' and i<2:
                                obs=Observations(**p['observations'])
                                with torch.no_grad():out=sample(model,p['cache'],obs,steps=500,seed=41000+i)
                                f=p['cache']['conditioning']['feats'];n=p['labels'].shape[1]
                                m=f['token_to_bb4_atoms'].float().reshape(1,n*4,-1)
                                bb=torch.bmm(m,out['coordinates'])[0].reshape(n,4,3).cpu().numpy()
                                true=torch.bmm(m,p['target'])[0].reshape(n,4,3).cpu().numpy()
                                d=np.linalg.norm(np.diff(bb[:,1],axis=0),axis=-1)
                                rec['max_ca_distance']=float(d.max())
                                if step==0:
                                    assert np.isfinite(bb).all() and d.max()<20., 'Catastrophic full-generation failure before training'
                                    assert out['feedback_calls']==500 and out['native_sampler']
                                rec.update(ca_lddt=ca_lddt(bb[:,1],true[:,1]),abnormal_ca_distances=int(((d<3.4)|(d>4.2)).sum()),finite_geometry=bool(np.isfinite(bb).all()))
                            rows.append(rec);del p,parts
                save_json(RUN/f'validation-{step}.json',rows)
                values={}
                for split in ('train','validation'):
                    for mode in ('folding','design'):
                        chosen=[r for r in rows if r['split']==split and r['mode']==mode]
                        for k in ('coordinate','smooth_lddt','identity','replay','ca_lddt','abnormal_ca_distances'):
                            vals=[r[k] for r in chosen if k in r]
                            if vals:values[f'validation/{split}_probe/{mode}/{k}']=float(np.mean(vals))
                tracker.log(values,step);print('VALIDATION',step,json.dumps(values),flush=True)
            finally:
                random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3]);model.train()
        if not resume: validate(0,rollout=True)
        del first
        random.seed(1729);np.random.seed(1729);torch.manual_seed(1729)
        orders={'folding':[],'design':[]}
        if resume:
            assert resume['config']['identity_weight'] == weight
            optimizer.load_state_dict(resume['optimizer'])
            orders=resume['orders']; completed_step=resume['step']
            random.setstate(resume['rng'][0]); np.random.set_state(resume['rng'][1])
            torch.set_rng_state(resume['rng'][2]); torch.cuda.set_rng_state_all(resume['rng'][3])
            tracker.log({'resume/from_completed_update':completed_step},completed_step)
            del resume
        def save_resume():
            state=dict(adapter=model.trainable_state(), optimizer=optimizer.state_dict(),
                step=completed_step, orders=orders, config=cfg,
                architecture='native-condition-full-diffusion-no-lora', shared_across_tasks=True,
                rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()))
            temp=RUN/'resume.tmp';torch.save(state,temp);temp.replace(resume_path)
        for step in range(completed_step+1,cfg['steps']+1):
            optimizer.zero_grad(set_to_none=True);logs={};meta_logs=[]
            for micro in range(cfg['gradient_accumulation']):
                mode=('folding','design')[micro%2]
                if not orders[mode]:orders[mode]=random.sample(data['train'],len(data['train']))
                row=orders[mode].pop();p=load(row,mode,'train')
                parts,meta=losses(model,p);geometry=parts['coordinate']+parts['smooth_lddt']
                objective=geometry+weight*parts['identity']+parts['replay']
                assert torch.isfinite(objective)
                if micro<2 and (step==1 or step%25==0):
                    for key, value in [('geometry',geometry),('identity',weight*parts['identity']),('replay',parts['replay'])]:
                        logs[f'train/{mode}/gradient_{key}']=gradient_norm(value,shared)
                for key,value in {**parts,'loss':objective,'weighted_identity':weight*parts['identity']}.items():
                    name=f'train/{mode}/{key}';logs[name]=logs.get(name,0)+float(value.detach())/(cfg['gradient_accumulation']/2)
                (objective/cfg['gradient_accumulation']).backward();meta_logs.append(meta)
                del p,parts,geometry,objective
            grad_before=float(torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True))
            for group in optimizer.param_groups:
                group['lr']=group['peak_lr']*min(step/cfg['warmup'],1.)
            if step==1:save_json(RUN/'started_updates.json',dict(first_step=1))
            optimizer.step()
            completed_step=step
            logs.update({'train/loss':(logs['train/folding/loss']+logs['train/design/loss'])/2,
                'train/gradient_norm_before_clip':grad_before,'train/gradient_norm_after_clip':min(grad_before,cfg['gradient_clip']),
                'train/gradient_clip_factor':min(1.,cfg['gradient_clip']/max(grad_before,1e-12)),
                'train/diffusion_lr':optimizer.param_groups[0]['lr'],'train/adapter_lr':optimizer.param_groups[1]['lr'],
                'train/noise_sigma_mean':float(np.mean([m['sigma'] for m in meta_logs])),
                'train/chemical_t_mean':float(np.mean([m['chemical_t'] for m in meta_logs])),
                'compute/GPU_hours':(time.monotonic()-start)/3600,'compute/peak_memory_GB':torch.cuda.max_memory_allocated()/1e9})
            tracker.log(logs,step)
            if step%10==0 or step==1:print('TRAIN',step,json.dumps(logs),flush=True)
            if step%100==0:
                save_resume()
                validate(step,rollout=step in (200,400))
                state=dict(adapter=model.trainable_state(),step=step,seed=1729,config=cfg,architecture='native-condition-full-diffusion-no-lora',shared_across_tasks=True)
                path=RUN/('final.pt' if step==cfg['steps'] else 'last.pt')
                torch.save(state,path)
                if step==cfg['steps']:tracker.checkpoint(path,selected=True)
                del state
        # A failure during final validation may leave a valid update-400 resume.
        if not (RUN/'final.pt').exists():
            assert completed_step==cfg['steps']
            validate(completed_step,rollout=True)
            torch.save(dict(adapter=model.trainable_state(),step=completed_step,seed=1729,config=cfg,
                architecture='native-condition-full-diffusion-no-lora',shared_across_tasks=True),RUN/'final.pt')
            tracker.checkpoint(RUN/'final.pt',selected=True)
        save_json(RUN/'completion.json',dict(steps=cfg['steps'],seconds=time.monotonic()-start,training_completed=True))
        import wandb
        artifact=wandb.Artifact('native-condition-training-audit',type='diagnostics')
        for p in RUN.glob('*.json'):artifact.add_file(str(p))
        artifact.add_file(str(DATA/'data_manifest.json'));tracker.run.log_artifact(artifact)
        ok=True
    except BaseException:
        if 'save_resume' in locals() and completed_step>0:
            save_resume()
        raise
    finally:tracker.finish(ok)


if __name__=='__main__':main()
