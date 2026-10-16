"""Validation-selected overnight fusion experiment, immutable data and diffusion."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import traceback
import torch
from common import ROOT, CODE, CONFIG, SOURCE, CKPT, Tracker, atomic_json, save_tensor, sha, seed, manifest
from learning import build_model, load_input, microbatches
from native_checks import state_hash, native_loss
from preflight import run as gpu_preflight
from holdout_eval import evaluate as holdout_eval

spec = importlib.util.spec_from_file_location('reference_training', CODE/'reference_train.py')
reference_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_training)
validation_eval = reference_training.validation_eval
MODALITIES = ['peptide', 'antibody', 'ligand']


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    seed(CONFIG['seed'])
    assert os.environ['FUSION_INTERFACE'] == CONFIG['interface']
    assert sha(ROOT/'manifests/manifest.json') == CONFIG['manifest_sha256']
    assert sha(CKPT) == CONFIG['anew_checkpoint_sha256']
    groups = manifest()
    assert all(len(groups['train'][m]) == 10000 for m in MODALITIES)
    assert [len(groups['validation'][m]) for m in MODALITIES] == [114, 400, 100]
    out = ROOT/'training'
    out.mkdir(exist_ok=True)
    assert not (out/'STARTED.json').exists(), 'Do not repeat or extend a started budget without an audit'
    model, fusion, epsilon, frozen = build_model()
    initial_hash = state_hash(fusion.state_dict())
    base_hash = state_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})
    trainable = [p for p in model.parameters() if p.requires_grad]
    param_groups = [{'params':list(fusion.parameters()),'lr':CONFIG['lr'],'name':'fusion'}]
    if CONFIG['train_backbone']:
        param_groups.append({'params':list(epsilon.base.parameters()),'lr':CONFIG['backbone_lr'],'name':'native_epsilon'})
    assert len({id(p) for g in param_groups for p in g['params']}) == len(trainable)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
    provenance = dict(config=CONFIG, initial_adapter_state_hash=initial_hash, frozen_base_hash=base_hash,
                      trainable_parameters=sum(p.numel() for p in trainable), fusion_parameters=sum(p.numel() for p in fusion.parameters()), gpu=torch.cuda.get_device_name(),
                      source_hashes={p.name:sha(p) for p in CODE.glob('*.py')},
                      manifest_sha256=CONFIG['manifest_sha256'], native_forward_sha256=sha(SOURCE/'models/LDM/ldm_clean.py'),
                      budget=str(CONFIG['budget_seconds'])+' seconds on one GPU from job startup; generation evaluation recorded separately',
                      checkpoint_selection='Minimum unchanged 614-record native validation total among trained checkpoints; one shared checkpoint',
                      benchmark_used_for_selection=False)
    tracker = Tracker('training-native', provenance)
    tracker.run.define_metric('*', hidden=True)
    tracker.run.define_metric('optimizer_step', hidden=True)
    for pattern in ['train/*','optim/*','system/*','validation/*','progress/*']:
        tracker.run.define_metric(pattern, step_metric='optimizer_step', hidden=True)
    tracker.run.define_metric('train/total', step_metric='optimizer_step', hidden=False)
    tracker.run.define_metric('optim/grad_norm', step_metric='optimizer_step', hidden=False)
    for modality in MODALITIES+['combined']:
        tracker.run.define_metric(f'validation/{modality}/*', step_metric='optimizer_step', hidden=False)
        tracker.run.define_metric(f'holdout/{modality}/*', step_metric='optimizer_step', hidden=False)
    atomic_json(out/'provenance.json', provenance)
    atomic_json(out/'STARTED.json', {'unix':time.time(), 'provenance':provenance})
    torch.cuda.synchronize()
    start = time.monotonic()
    prior_job_seconds = time.time()-float(os.environ['FUSION_BUDGET_START_UNIX'])
    deadline = start + CONFIG['budget_seconds']-prior_job_seconds-90.
    optimizer_seconds = 0.
    validation_seconds = 0.
    best = None
    records = []
    steps = presentations = 0
    modality_presentations = dict.fromkeys(MODALITIES, 0)
    largest_step_seconds = 5.
    next_validation = 500
    validation_calls = 0

    def validate(initial=False):
        nonlocal validation_seconds, validation_calls, best
        tick = time.monotonic()
        val = validation_eval(model, epsilon, groups['validation'])
        torch.cuda.synchronize()
        duration = time.monotonic()-tick
        validation_seconds += duration
        validation_calls += 1
        record = dict(optimizer_step=steps, presentations=presentations, elapsed_seconds=time.monotonic()-start,
                      validation_seconds=duration, validation=val)
        records.append(record)
        tracker.log({'optimizer_step':steps,
                     **{f'validation/{m}/{k.lower()}':v for m,values in val.items() for k,v in values.items()},
                     'system/budget_elapsed_seconds':time.monotonic()-start})
        if initial:
            atomic_json(out/'initial_validation.json',val)
        else:
            state = {'fusion':fusion.state_dict(), 'optimizer':optimizer.state_dict(), 'step':steps,
                     'presentations':presentations, 'provenance':provenance, 'interface':CONFIG['interface'],
                     'validation':val, 'elapsed_seconds':time.monotonic()-start}
            if CONFIG['train_backbone']:
                state['epsilon_base'] = epsilon.base.state_dict()
            save_tensor(out/'last.pt',state)
            if best is None or val['combined']['total'] < best['validation']['combined']['total']:
                save_tensor(out/'best_joint.pt',state)
                best = dict(record,checkpoint=str(out/'best_joint.pt'),checkpoint_sha256=sha(out/'best_joint.pt'))
                atomic_json(out/'selection.json',best)
        atomic_json(out/'metrics.json',records)
        print('VALIDATION',json.dumps(record),flush=True)
        eval_checkpoint=out/'evaluation_checkpoints'/f'step_{steps:06d}.pt'
        eval_state={'fusion':fusion.state_dict(),'step':steps,'validation':val,'provenance':provenance}
        if CONFIG['train_backbone']:eval_state['epsilon_base']=epsilon.base.state_dict()
        save_tensor(eval_checkpoint,eval_state)
        holdout=holdout_eval(model,epsilon,steps,eval_checkpoint,tracker)
        print('HOLDOUT',json.dumps({'optimizer_step':steps,'results':holdout['results'],'seconds':holdout['seconds']}),flush=True)
        return time.monotonic()-tick

    try:
        preflight = gpu_preflight(model,fusion,epsilon,frozen,groups,tracker)
        seed(CONFIG['seed'])
        last_validation_duration = validate(initial=True)
        all_rows = sum(groups['train'].values(), [])
        epoch = 0
        stop = False
        while not stop:
            order = list(all_rows)
            random.Random(CONFIG['seed']+epoch).shuffle(order)
            for offset in range(0,len(order),CONFIG['effective_batch']):
                assert not (ROOT.parent/'USER_STOPPED.json').exists()
                # Reserve measured validation time plus a margin, and never begin
                # a step expected to cross the training/validation budget.
                reserve = max(30., last_validation_duration*1.5+10.)
                if time.monotonic()+reserve+largest_step_seconds*1.5 >= deadline:
                    stop = True
                    break
                tick = time.monotonic()
                seed(CONFIG['seed']+steps)
                model.train()
                model.autoencoder.eval()
                group = order[offset:offset+CONFIG['effective_batch']]
                n_total = sum(int(load_input(r)['generate_mask'].sum()) for r in group)
                optimizer.zero_grad(set_to_none=True)
                sums = dict(H=0.,X=0.,total=0.)
                details = {}
                for modality in MODALITIES:
                    subset = [r for r in group if r['modality']==modality]
                    ms = dict(H=0.,X=0.,total=0.)
                    mn = 0
                    for micro in microbatches(subset,max_cost=70000,max_count=2):
                        values,n = native_loss(model,epsilon,micro)
                        (values['total']*(n/n_total)).backward()
                        mn += n
                        for k in sums:
                            v = float(values[k].detach())
                            sums[k] += v*n/n_total
                            ms[k] += v*n
                    if mn:
                        details.update({f'train/{modality}/{k.lower()}':v/mn for k,v in ms.items()})
                    modality_presentations[modality] += len(subset)
                assert all(p.grad is None for p in frozen)
                norm = torch.nn.utils.clip_grad_norm_(trainable,CONFIG['grad_clip'],error_if_nonfinite=True)
                progress = min(1., (time.monotonic()-start)/max(1.,deadline-start))
                warm = min(1., (steps+1)/100.)
                factor = warm*(.2+.8*.5*(1+__import__('math').cos(__import__('math').pi*progress)))
                for pg in optimizer.param_groups:
                    pg['lr'] = (CONFIG['lr'] if pg['name']=='fusion' else CONFIG['backbone_lr'])*factor
                optimizer.step()
                torch.cuda.synchronize()
                duration = time.monotonic()-tick
                optimizer_seconds += duration
                largest_step_seconds = max(largest_step_seconds,duration)
                steps += 1
                presentations += len(group)
                tracker.log({'optimizer_step':steps,'progress/epoch':epoch+1,'progress/presentations':presentations,
                             **{'train/'+k.lower():v for k,v in sums.items()},**details,
                             'optim/grad_norm':float(norm),'optim/lr':optimizer.param_groups[0]['lr'],
                             'optim/backbone_lr':optimizer.param_groups[-1]['lr'] if CONFIG['train_backbone'] else 0.,
                             'system/step_seconds':duration,'system/optimizer_seconds':optimizer_seconds,
                             'system/budget_elapsed_seconds':time.monotonic()-start,
                             'system/gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30})
                if steps%25==0:
                    print('TRAIN',CONFIG['interface'],steps,presentations,round(time.monotonic()-start,1),flush=True)
                if steps >= next_validation:
                    last_validation_duration = validate()
                    next_validation += 500
            epoch += 1
        if steps and (best is None or records[-1]['optimizer_step'] != steps):
            last_validation_duration = validate()
        torch.cuda.synchronize()
        budget_elapsed = time.monotonic()-start
        assert steps>0 and best is not None
        assert state_hash({n:p for n,p in model.named_parameters() if not p.requires_grad}) == base_hash
        report = dict(best, completed=True, optimizer_steps=steps, presentations=presentations,
                      modality_presentations=modality_presentations, training_validation_seconds=budget_elapsed,
                      job_elapsed_seconds_before_bookkeeping=prior_job_seconds+budget_elapsed,
                      gpu_preflight_seconds=None,
                      previous_failed_allocation_seconds=CONFIG.get('previous_failed_allocation_seconds',0),
                      total_setting_seconds_before_bookkeeping=CONFIG.get('previous_failed_allocation_seconds',0)+prior_job_seconds+budget_elapsed,
                      preflight_startup_seconds=prior_job_seconds,
                      budget_seconds=CONFIG['budget_seconds'],budget_exceeded=prior_job_seconds+budget_elapsed>CONFIG['budget_seconds'],
                      optimizer_seconds=optimizer_seconds,validation_seconds=validation_seconds,
                      validation_calls=validation_calls,gpu_count=1,gpu=torch.cuda.get_device_name(),
                      initial_adapter_state_hash=initial_hash,frozen_parameters_unchanged=True,
                      base_unchanged=not CONFIG['train_backbone'],trainable_parameters=sum(p.numel() for p in trainable),
                      fusion_parameters=sum(p.numel() for p in fusion.parameters()),
                      selected_checkpoint_is_shared=True,benchmark_complete=False)
        # An unexpected machine stall is disclosed as a budget failure, not hidden.
        atomic_json(out/'RESULT.json',report)
        assert not report['budget_exceeded'], 'Wall-clock budget exceeded; result must not be called equal-compute'
        tracker.finish(report,[out/'best_joint.pt',out/'selection.json',out/'metrics.json',out/'provenance.json',out/'RESULT.json'])
        atomic_json(out/'COMPLETE.json',report)
        print('COMPLETE',json.dumps(report),flush=True)
    except BaseException as exc:
        failure = {'error':str(exc),'traceback':traceback.format_exc(),'optimizer_steps':steps,
                   'elapsed_seconds':time.monotonic()-start,'interface':CONFIG['interface']}
        atomic_json(out/'FAILED.json',failure)
        tracker.run.summary.update(failure)
        tracker.run.finish(exit_code=1)
        raise


if __name__=='__main__':
    main()
