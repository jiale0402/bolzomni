"""Exercise optimizer updates for every observation pattern before the full run."""
import json
import time
import shutil
import torch
from poc_model import ROOT,load_model,move
from model import OmniCompletion,Observations,completion_loss

EXP=ROOT/'experiments/omni_joint'

def main():
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    torch.manual_seed(1729)
    rows=json.loads((EXP/'manifests/cache.json').read_text())
    rows=[r for r in rows if r['split']=='train']
    base=load_model(); model=None; measurements=[]
    for row in rows:
        payload=move(torch.load(row['path'],map_location='cpu',weights_only=False),'cuda')
        cache=payload['cache']
        if model is None:
            model=OmniCompletion(base,cache['conditioning']['s_trunk'].shape[-1],cache['z_trunk'].shape[-1],
                                 base.structure_module.score_model.a_norm.normalized_shape[0]).cuda()
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        torch.cuda.synchronize(); started=time.monotonic()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss,parts=completion_loss(model,cache,payload['target'],payload['labels'],payload['resolved'],
                                       Observations(**payload['observations']))
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step()
        torch.cuda.synchronize()
        measurements.append(dict(mode=row['mode'],seconds_per_step=(time.monotonic()-started)/2,
                                 loss=float(loss.detach()),tokens=int(payload['labels'].shape[1])))
        print('PROFILE',measurements[-1],flush=True)
    assert {r['mode'] for r in measurements}=={'folding','design','inverse','mixed'}
    report=dict(measurements=measurements,peak_memory_GB=torch.cuda.max_memory_allocated()/1e9,
                gpu=torch.cuda.get_device_name(),optimizer_updates_are_diagnostic=True)
    data=json.loads((EXP/'manifests/data_filtered.json').read_text())
    cache_profile=json.loads((EXP/'manifests/cache_profile.json').read_text())
    all_rows=[r for split in ('train','validation') for r in data[split]]
    sampled=[r for split in ('train','validation') for r in data[split][:2]]
    scale=sum(r['length']**2 for r in all_rows)/sum(r['length']**2 for r in sampled)
    estimated_bytes=cache_profile['bytes']*scale
    timings=cache_profile['measurements']
    assert timings,'Profile must rebuild a small cache to measure prompt throughput'
    total_square=4*sum(r['length']**2 for r in all_rows)
    estimated_cache_hours=sum(r['seconds']/r['tokens']**2 for r in timings)/len(timings)*total_square/3600
    # Quadratic length scaling is conservative for pair tensors; forecasts are
    # estimates, not a promised completion time. Slurm enforces walltime bounds.
    mean_square=sum(r['length']**2 for r in data['train'])/len(data['train'])
    step_seconds=sum(r['seconds_per_step']*mean_square/r['tokens']**2 for r in measurements)/len(measurements)
    report.update(estimated_cache_GB=estimated_bytes/1e9,estimated_cache_hours=estimated_cache_hours,
                  estimated_training_hours=step_seconds*6500/3600,
                  free_disk_GB=shutil.disk_usage(EXP).free/1e9)
    report['within_budget']=bool(estimated_bytes<shutil.disk_usage(EXP).free*0.85 and
                                estimated_cache_hours<6 and report['estimated_training_hours']<6)
    (EXP/'results/training_profile.json').write_text(json.dumps(report,indent=2))
    print('PROFILE_REPORT',json.dumps(report),flush=True)
    assert report['within_budget'],'Measured footprint exceeds allocated budget; optimize caching before full run'

if __name__=='__main__':main()
