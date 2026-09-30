"""One continuation of existing Omni; requires explicit additional compute approval.

No native-architecture control training. Historical checkpoints/source are read-only.
"""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--authorized-continuation', action='store_true')
    args = ap.parse_args()
    if not args.authorized_continuation or not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Requires separately approved Quest continuation through Slurm')
    import numpy as np
    import torch
    from poc_model import ROOT as PROJECT, ASSETS, load_model, move
    from noise_matched_flow import FullOmni, losses, native_observations, refresh_teacher_bridge
    from chemical_schedule import joint_objective
    from tracking import Tracker, file_hash

    ROOT = Path(__file__).resolve().parent
    DATA = ROOT.parent/'official_poc'
    PRIOR = ROOT.parent/'native_condition_poc/runs/native-1729'
    RUN = ROOT/'runs/continue-1729'
    RUN.mkdir(parents=True, exist_ok=True)
    assert not (RUN/'completion.json').exists(), 'Completed continuation cannot restart'
    assert not (RUN/'interrupted_optimizer_step.json').exists(), 'Partial optimizer update requires explicit recovery review'
    resume_path = RUN/'resume.pt'
    if (RUN/'started_updates.json').exists() and not resume_path.exists():
        raise RuntimeError('Updates were started without recoverable state; never restart from step400 silently')
    source_path = resume_path if resume_path.exists() else PRIOR/'resume.pt'
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    continued_here = source_path == resume_path
    assert (400 <= source['step'] <= 2000) if continued_here else (source['step'] == 400)
    expected_final = 'e7d991ab7ecf0e88c78f2476ff54eeaa03c871af5ffff7a88c2ac0bae819c725'
    assert file_hash(PRIOR/'final.pt') == expected_final
    data = json.loads((DATA/'data_manifest.json').read_text())
    assert len(data['train']) == 128 and len(data['validation']) == 16
    assert all(r['source'] == 'boltzgen/boltzgen1_train' for r in data['train'])
    assert file_hash(DATA/'data_manifest.json') == source['config']['data_manifest_sha256']
    assert file_hash(ASSETS/'boltzgen1_diverse.ckpt') == source['config']['base_checkpoint_sha256']
    cfg = dict(source['config'])
    cfg.update(start_step=400, steps=2000, additional_updates=1600,
        source_checkpoint_sha256=expected_final, source_optimizer_sha256=file_hash(PRIOR/'resume.pt'),
        chemical_schedule='coordinate_noise_cdf', replay_weight=1.,
        checkpoint_selection='fixed final total update 2000, shared between both tasks; no test-based selection',
        actual_update_probe='before/after identical 8 microbatches with exact RNG restore; first and every100 updates',
        hard_stage_minutes=90, cumulative_stage_budget_gpu_hours=2.,
        training_control=False, calibration_repeated=False, GPU=torch.cuda.get_device_name(),
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    pinned = [ROOT/'chemical_schedule.py', ROOT/'noise_matched_flow.py', Path(__file__),
        ROOT.parent/'native_condition_poc/native_flow.py',
        PROJECT/'experiments/omni_joint/model.py', PROJECT/'experiments/omni_joint/full_model.py',
        PROJECT/'experiments/official_poc/corrected_model.py']
    cfg['continuation_sources'] = {str(p.relative_to(PROJECT)):file_hash(p) for p in pinned}
    for path in pinned[3:]:
        key=str(path.relative_to(PROJECT))
        assert file_hash(path) == source['config']['sources'][key], 'Historical model/loss source drift'
    if continued_here:
        assert cfg['continuation_sources'] == source['config']['continuation_sources'], 'Do not modify live/resumed source'
    (RUN/'config.json').write_text(json.dumps(cfg, indent=2))
    tracker = Tracker(RUN, cfg, 'noise-matched-omni-continuation')
    journal = (RUN/'microbatches.jsonl').open('a', buffering=1)
    started = time.monotonic(); completed = int(source['step']); success = False
    update_in_progress = False; boundary_rng = None; boundary_orders = None; active_step = None
    # Slurm's early warning requests a checkpoint at an optimizer boundary. It
    # never interrupts optimizer.step halfway through a parameter update.
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    def save_json(path, value):
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)); temp.replace(path)
    def rng_state():
        return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    def restore(state):
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])
    try:
        teacher, base = load_model(), load_model()
        def load(row, mode):
            path = DATA/'cache'/f'{row["id"]}-{mode}.pt'
            assert path.exists(), 'No undeclared data rebuilding during continuation'
            payload = move(torch.load(path,map_location='cpu',weights_only=False),'cuda')
            assert payload['hidden_coordinate_audit']
            refresh_teacher_bridge(teacher,payload['cache'])
            return native_observations(payload)
        first = load(data['train'][0],'design')
        model = FullOmni(base,teacher,first['cache']).cuda()
        assert set(source['adapter']) == {n for n,p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(source['adapter'], strict=False)
        del first
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in parameters) == cfg['trainable_parameters']
        native = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')]
        new = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('adapter.')]
        optimizer = torch.optim.AdamW([{'params':native,'lr':cfg['diffusion_lr'],'peak_lr':cfg['diffusion_lr']},
            {'params':new,'lr':cfg['new_module_lr'],'peak_lr':cfg['new_module_lr']}],
            betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
        optimizer.load_state_dict(source['optimizer'])
        # Preserve actual optimizer, LR, ordering and RNG; no repeat warmup/reset.
        assert optimizer.param_groups[0]['lr'] == cfg['diffusion_lr']
        assert optimizer.param_groups[1]['lr'] == cfg['new_module_lr']
        orders = source['orders']; prior_rng = source['rng']
        del source
        def objective(parts):
            return joint_objective(parts,identity_weight=cfg['identity_weight'],replay_weight=cfg['replay_weight'])
        def validate(step):
            saved = rng_state(); rows=[]; model.eval()
            try:
                for split in ('train','validation'):
                    for i,row in enumerate(data[split][:4 if split=='train' else 16]):
                        for mode in ('folding','design'):
                            payload=load(row,mode)
                            with torch.no_grad(): parts,meta=losses(model,payload,seed=31000+i,
                                sigma_override=8.,chemical_t_override=.5)
                            rows.append(dict(split=split,id=row['id'],mode=mode,
                                **{k:float(v) for k,v in parts.items()},objective=float(objective(parts)),**meta))
                            del payload,parts
                save_json(RUN/f'validation-{step}.json',rows)
                logs={}
                for split in ('train','validation'):
                    for mode in ('folding','design'):
                        chosen=[r for r in rows if r['split']==split and r['mode']==mode]
                        for k in ('coordinate','smooth_lddt','identity','replay','objective'):
                            logs[f'validation/{split}_probe/{mode}/{k}']=float(np.mean([r[k] for r in chosen]))
                for split in ('train','validation'):
                    logs[f'validation/{split}_probe/total']=(logs[f'validation/{split}_probe/folding/objective']+
                        logs[f'validation/{split}_probe/design/objective'])/2
                tracker.log(logs,step)
            finally:
                model.live=None; restore(saved); model.train()
        def save_resume():
            obj=dict(adapter=model.trainable_state(),optimizer=optimizer.state_dict(),step=completed,
                orders=orders,rng=rng_state(),config=cfg,architecture='noise-matched-full-diffusion-no-lora',
                shared_across_tasks=True)
            temp=RUN/'resume.tmp'; torch.save(obj,temp); temp.replace(resume_path)
        restore(prior_rng); model.train()
        if not continued_here:
            validate(400)
            tracker.log({'resume/from_completed_update':400},400)
            save_resume()
        for step in range(completed+1,cfg['steps']+1):
            if stop_requested:
                save_resume()
                raise TimeoutError('Slurm time warning: preserved checkpoint; no automatic resubmission')
            active_step = step
            boundary_rng = rng_state()
            boundary_orders = {mode:list(rows) for mode,rows in orders.items()}
            optimizer.zero_grad(set_to_none=True); logs={}; packets=[]
            inspect_update = step==401 or step%100==0
            for micro in range(cfg['gradient_accumulation']):
                mode=('folding','design')[micro%2]
                if not orders[mode]: orders[mode]=random.sample(data['train'],len(data['train']))
                row=orders[mode].pop(); payload=load(row,mode)
                before_rng=rng_state() if inspect_update else None
                parts,meta=losses(model,payload); value=objective(parts)
                assert torch.isfinite(value), 'Non-finite training objective'
                values={k:float(v.detach()) for k,v in parts.items()}
                values.update(objective=float(value.detach()),geometry=values['coordinate']+values['smooth_lddt'])
                for key,v in values.items():
                    name=f'train/{mode}/{key}';logs[name]=logs.get(name,0)+v/(cfg['gradient_accumulation']/2)
                journal.write(json.dumps(dict(step=step,micro=micro,id=row['id'],mode=mode,**meta,**values))+'\n')
                if inspect_update: packets.append((row,mode,before_rng,values,meta))
                (value/cfg['gradient_accumulation']).backward()
                del payload,parts,value
                model.live=None
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True))
            if step==401: save_json(RUN/'started_updates.json',dict(first_step=401,source_step=400))
            update_in_progress = True
            optimizer.step(); completed=step
            update_in_progress = False
            logs.update({'train/loss':(logs['train/folding/objective']+logs['train/design/objective'])/2,
                'train/gradient_norm_before_clip':norm,'train/gradient_clip_factor':min(1.,cfg['gradient_clip']/max(norm,1e-12)),
                'train/diffusion_lr':optimizer.param_groups[0]['lr'],'train/adapter_lr':optimizer.param_groups[1]['lr'],
                'compute/process_GPU_hours':(time.monotonic()-started)/3600})
            if inspect_update:
                after_rng=rng_state(); checks=[]
                try:
                    with torch.no_grad():
                        for row,mode,state,before,before_meta in packets:
                            payload=load(row,mode); restore(state)
                            parts,meta=losses(model,payload)
                            for key in ('sigma','chemical_t','masked_identities'):
                                assert meta[key] == before_meta[key], 'Actual update probe RNG mismatch'
                            after={k:float(v) for k,v in parts.items()}
                            after.update(objective=float(objective(parts)),geometry=after['coordinate']+after['smooth_lddt'])
                            checks.append(dict(id=row['id'],mode=mode,before=before,after=after,
                                delta={k:after[k]-before[k] for k in before},**meta))
                            del payload,parts
                    save_json(RUN/f'actual-update-{step}.json',checks)
                    for mode in ('folding','design'):
                        selected=[r for r in checks if r['mode']==mode]
                        for key in ('geometry','objective','identity','replay'):
                            logs[f'actual_update/{mode}/{key}_delta']=float(np.mean([r['delta'][key] for r in selected]))
                finally: model.live=None; restore(after_rng)
            tracker.log(logs,step)
            if step==401 or step%50==0: print('TRAIN',step,json.dumps(logs),flush=True)
            if step==401 or step%100==0:
                save_resume()
            if step%200==0: validate(step)
        assert completed==2000
        final=RUN/'final.pt'
        torch.save(dict(adapter=model.trainable_state(),step=completed,config=cfg,
            architecture='noise-matched-full-diffusion-no-lora',shared_across_tasks=True),final)
        tracker.checkpoint(final,selected=True)
        save_json(RUN/'completion.json',dict(steps=completed,additional_updates=1600,
            seconds=time.monotonic()-started,final_sha256=file_hash(final),training_completed=True))
        import wandb
        artifact = wandb.Artifact('noise-matched-continuation-records', type='diagnostics')
        journal.flush()
        for path in list(RUN.glob('*.json')) + [RUN/'microbatches.jsonl', RUN/'metrics.jsonl',
                DATA/'data_manifest.json', ROOT/'PLAN.md', *pinned]:
            artifact.add_file(str(path), name=str(path.relative_to(PROJECT)))
        tracker.run.log_artifact(artifact)
        success=True
    except BaseException:
        if update_in_progress:
            save_json(RUN/'interrupted_optimizer_step.json',dict(step=active_step,
                last_fully_completed_step=completed,automatic_resume_forbidden=True))
        elif 'save_resume' in locals():
            if active_step is not None and completed < active_step:
                # Failed before an update: restore data/RNG to the last boundary.
                orders = boundary_orders; restore(boundary_rng)
            save_resume()
        raise
    finally:
        journal.close(); tracker.finish(success)


if __name__=='__main__':
    main()
