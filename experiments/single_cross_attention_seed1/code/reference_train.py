"""One shared adapter, unchanged Anew forward/loss, 30k mixed logical samples."""
import argparse
import copy
import json
import random
import time
import torch
from common import ROOT, CODE, CONFIG, SOURCE, CKPT, seed, sha, atomic_json, save_tensor, Tracker, manifest
from learning import build_model, materialize, load_input, features
from native_checks import (state_hash, rng_state, restore_rng, fixed_rng, evaluate,
    optimize_step, native_loss, loss_formula_check, bypass_check)


def profile(model,fusion,epsilon,frozen,rows):
    pristine=copy.deepcopy(fusion.state_dict())
    base_hash=state_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})
    report={}
    model.eval()
    try:
        for modality in ['peptide','antibody','ligand']:
            sample=[r for r in rows if r['modality']==modality]
            assert sample
            row=min(sample,key=lambda r:abs(r['nblocks']-100))
            formula=loss_formula_check(model,epsilon,[row])
            bypass=bypass_check(model,epsilon,[row])
            with torch.no_grad():fusion.output.weight.normal_(std=.001)
            f=features([row]);mask=f[0]['mask'];gen=f[0]['generate_mask']
            c=torch.randn(len(mask),512,device='cuda',requires_grad=True)
            update=fusion(c,f)
            assert torch.count_nonzero(update[~mask])==0
            if modality=='ligand':assert not bool(mask[gen].any())
            fusion.zero_grad(set_to_none=True)
            with fixed_rng(4712):values,_=native_loss(model,epsilon,[row])
            values['total'].backward()
            grad={n:float(p.grad.abs().sum()) for n,p in fusion.named_parameters() if p.grad is not None}
            assert grad and all(torch.isfinite(torch.tensor(v)) for v in grad.values()) and sum(grad.values())>0,(modality,grad)
            assert all(p.grad is None for p in frozen)
            # Same native epsilon arguments, changing only donor features. This checks
            # context -> generated outputs without a second VAE/noise realization.
            capture={}
            h=epsilon.register_forward_pre_hook(lambda m,a:capture.update(args=a))
            with torch.no_grad(),fixed_rng(4712):native_loss(model,epsilon,[row])
            h.remove()
            epsilon.set_features(f)
            with torch.no_grad():
                out1=epsilon(*capture['args'])
                epsilon.set_features([{**v,'single':torch.zeros_like(v['single']),'pair':torch.zeros_like(v['pair'])} for v in f])
                out2=epsilon(*capture['args'])
            effect=sum(float((a[gen]-b[gen]).abs().sum()) for a,b in zip(out1,out2))
            assert effect>1e-6,(modality,'donor has no generated-state effect')
            report[modality]={'native_loss_formula':formula,'bypass':bypass,'gradient_l1':grad,
                'generated_output_donor_effect_l1':effect,'uncovered_direct_delta_exact_zero':True,
                'coverage':int(mask.sum()),'blocks':len(mask),'covered_generated':int(mask[gen].sum())}
            fusion.load_state_dict(pristine);fusion.zero_grad(set_to_none=True)
        # Largest selected sample in each modality must also run backward.
        for modality in report:
            row=max([r for r in rows if r['modality']==modality],key=lambda r:r['nblocks'])
            with fixed_rng(40001):values,_=native_loss(model,epsilon,[row])
            values['total'].backward();fusion.zero_grad(set_to_none=True)
        assert state_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})==base_hash
        report['passed']=True
        report['gpu']=torch.cuda.get_device_name()
        report['peak_gib']=torch.cuda.max_memory_allocated()/2**30
        atomic_json(ROOT/'profile/PASSED.json',report)
        return report
    finally:
        fusion.load_state_dict(pristine);fusion.zero_grad(set_to_none=True)
        epsilon.gamma=1


def validation_eval(model,epsilon,groups):
    results={}
    for i,(modality,rows) in enumerate(groups.items()):
        with fixed_rng(6729000+i):selected=materialize(rows)
        results[modality]=evaluate(model,epsilon,selected,7729000+i*10000)
    # Same generated-block reduction as the native combined validation objective.
    denominator=sum(v['generated_residues'] for v in results.values())
    results['combined']={k:sum(v[k]*v['generated_residues'] for v in results.values())/denominator for k in ['H','X','total']}
    return results


def flatten(prefix,groups):
    return {prefix+'/'+modality+'/'+k:v for modality,values in groups.items() for k,v in values.items()}


def validation_logs(groups):
    return {'validation/'+m+'/'+k.lower():groups[m][k]
            for m in ['peptide','antibody','ligand'] for k in ['total','X','H']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');seed(CONFIG['seed'])
    m=manifest()
    assert all(len(m['train'][k])==10000 for k in ['peptide','antibody','ligand'])
    assert not (ROOT/'training/COMPLETE.json').exists(),'already complete'
    assert not (ROOT/'USER_STOPPED.json').exists()
    model,fusion,epsilon,frozen=build_model()
    assert sha(CKPT)==CONFIG['anew_checkpoint_sha256']
    base_hash=state_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})
    provenance={'config':CONFIG,'manifest_sha256':sha(ROOT/'manifests/manifest.json'),
        'anew_checkpoint_sha256':sha(CKPT),'frozen_base_sha256':base_hash,
        'source_hashes':{p.name:sha(p) for p in CODE.glob('*.py')},
        'native_forward_sha256':sha(SOURCE/'models/LDM/ldm_clean.py'),
        'native_loss':'model(**native_batch); H*h_loss_weight+X; generated-block reduction unchanged',
        'h_loss_weight':model.h_loss_weight,'condition_sampler':model.use_condtion_sampler,
        'is_aa_corrupt_ratio':model.is_aa_corrupt_ratio,'latent_deterministic':model.latent_deterministic,
        'trainable_parameters':sum(p.numel() for p in fusion.parameters()),'initialization':'fresh shared adapter'}
    tracker=Tracker('profile' if args.profile_only else 'training-native',provenance)
    if not args.profile_only:
        tracker.run.define_metric('optimizer_step')
        tracker.run.define_metric('train/*', step_metric='optimizer_step')
        tracker.run.define_metric('optim/*', step_metric='optimizer_step')
        tracker.run.define_metric('system/*', step_metric='optimizer_step')
        tracker.run.define_metric('progress/epoch')
        tracker.run.define_metric('validation/*', step_metric='progress/epoch')
    out=ROOT/'training';out.mkdir(exist_ok=True)
    if args.profile_only:
        report=profile(model,fusion,epsilon,frozen,json.loads((ROOT/'manifests/profile_rows.json').read_text()))
        tracker.finish(report,[ROOT/'profile/PASSED.json']);return
    assert (ROOT/'profile/PASSED.json').exists() and (ROOT/'DATA_VERIFIED.json').exists()
    atomic_json(out/'provenance.json',provenance)
    optimizer=torch.optim.AdamW(fusion.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    last=out/'last.pt';records=[];epoch0=0;step=0;gradients={}
    rows=sum(m['train'].values(),[])
    # Immutable 128-record training probes per modality; full validation retained.
    probes={k:random.Random(CONFIG['seed']+i).sample(v,128) for i,(k,v) in enumerate(m['train'].items())}
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        previous=copy.deepcopy(saved['provenance'])
        current=copy.deepcopy(provenance)
        previous_hashes=previous.pop('source_hashes');current_hashes=current.pop('source_hashes')
        assert previous==current, 'non-source provenance changed'
        assert set(previous_hashes)==set(current_hashes)
        assert all(previous_hashes[k]==current_hashes[k] for k in current_hashes if k not in ('train.py','common.py'))
        assert previous_hashes['common.py'] in ['cd00792b716488204c4072b294c900ac5bb7c23ce5247e8c27f1846c101a5755',current_hashes['common.py']], 'unapproved tracker source'
        assert previous_hashes['train.py'] in ['d1050bd0795c26c744539b09a0eacf58d27eb30340773a29aa0ffc9d18faf781',current_hashes['train.py']], 'unapproved training source'
        atomic_json(ROOT/'audits/logging_restart_20261008/resume.json',
            {'checkpoint':str(last),'epoch':saved['epoch'],'optimizer_step':saved['step'],
              'old_source_hash':previous_hashes['train.py'],'new_source_hash':current_hashes['train.py']})
        fusion.load_state_dict(saved['fusion']);optimizer.load_state_dict(saved['optimizer'])
        records=saved['records'];epoch0=saved['epoch'];step=saved['step'];gradients=saved['gradient_totals']
        restore_rng(saved['rng'])
        print('RESUMED',epoch0,step,flush=True)
    else:
        initial=validation_eval(model,epsilon,m['validation'])
        atomic_json(out/'initial_validation.json',initial)
        tracker.log({'progress/epoch':0,**validation_logs(initial)})
        save_tensor(out/'initial.pt',{'fusion':fusion.state_dict(),'provenance':provenance})
    for epoch in range(epoch0,CONFIG['epochs']):
        model.train();model.autoencoder.eval();seed(CONFIG['seed']+epoch)
        order=list(rows);random.Random(CONFIG['seed']+epoch).shuffle(order)
        totals={k:{'H':0.,'X':0.,'total':0.,'blocks':0,'records':0} for k in m['train']}
        tick=time.monotonic()
        for start in range(0,len(order),CONFIG['effective_batch']):
            assert not (ROOT/'USER_STOPPED.json').exists()
            step_started=time.monotonic()
            group=materialize(order[start:start+CONFIG['effective_batch']])
            # Accumulate native losses with their native generated-block weights.
            # Split by modality only to log components, not to introduce new weights.
            count=sum(int(load_input(r)['generate_mask'].sum()) for r in group)
            optimizer.zero_grad(set_to_none=True)
            step_losses={k:0. for k in ['H','X','total']}
            from learning import microbatches
            for modality in totals:
                for micro in microbatches([r for r in group if r['modality']==modality]):
                    values,n=native_loss(model,epsilon,micro)
                    (values['total']*(n/count)).backward()
                    for k in step_losses:step_losses[k]+=float(values[k].detach())*(n/count)
                    for k in ['H','X','total']:totals[modality][k]+=float(values[k].detach())*n
                    totals[modality]['blocks']+=n;totals[modality]['records']+=len(micro)
            assert all(p.grad is None for p in frozen)
            for name,p in fusion.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(),name
                    gradients[name]=gradients.get(name,0.)+float(p.grad.abs().sum())
            norm=torch.nn.utils.clip_grad_norm_(fusion.parameters(),CONFIG['grad_clip'],error_if_nonfinite=True)
            optimizer.step();step+=1
            tracker.log({'progress/epoch':epoch+1,'optimizer_step':step,
                'progress/records_in_epoch':start+len(group),
                'optim/grad_norm':float(norm),'optim/lr':optimizer.param_groups[0]['lr'],
                'system/step_seconds':time.monotonic()-step_started,
                'system/gpu_allocated_gib':torch.cuda.memory_allocated()/2**30,
                'system/gpu_reserved_gib':torch.cuda.memory_reserved()/2**30,
                'system/gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                **{'train/'+k.lower():v for k,v in step_losses.items()}})
            if step%20==0 or start==0:
                print('TRAIN',epoch+1,start+len(group),len(order),flush=True)
        val=validation_eval(model,epsilon,m['validation']);probe=validation_eval(model,epsilon,probes)
        record={'epoch':epoch+1,'epoch_seconds':time.monotonic()-tick,
            **flatten('validation',val),**flatten('fixed_train_probe',probe),
            **{f'train_stochastic/{m}/{k}':t[k]/t['blocks'] for m,t in totals.items() for k in ['H','X','total']}}
        records.append(record);tracker.log({'progress/epoch':epoch+1,**validation_logs(val)})
        state={'fusion':fusion.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
            'step':step,'records':records,'provenance':provenance,'gradient_totals':gradients,'rng':rng_state()}
        save_tensor(out/f'epoch_{epoch+1:02d}.pt',state);save_tensor(last,state)
        atomic_json(out/'metrics.json',records)
        print('EPOCH',json.dumps(record),flush=True)
    assert state_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})==base_hash
    best=min(records,key=lambda r:(r['validation/combined/total'],r['epoch']))
    checkpoint=out/f"epoch_{best['epoch']:02d}.pt"
    selected={'epoch':best['epoch'],'checkpoint':str(checkpoint),'sha256':sha(checkpoint),
        'criterion':'minimum combined native validation loss, generated-block weighted; earliest tie',
        'shared_across_modalities':True,'test_used':False,'validation':{k:v for k,v in best.items() if k.startswith('validation/')},
        'frozen_base_unchanged':True,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}
    atomic_json(out/'selection.json',selected)
    tracker.finish(selected,[checkpoint,out/'selection.json',out/'metrics.json',out/'provenance.json'])
    atomic_json(out/'COMPLETE.json',selected)


if __name__=='__main__':main()
