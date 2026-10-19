"""Exercise production batch-32 accumulation and largest cached records; no updates."""
import argparse
import json
import time
import torch
from common import ROOT,CONFIG,CKPT,Tracker,manifest,sha,seed,atomic_json
import learning
import native_checks as checks
from fusion import Fusion,FusionEpsilon
from training_batch import backward_batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');seed(42)
    stage='zero-update-stress-'+args.device
    dest=ROOT/'profile'/(stage+'.json')
    assert not dest.exists()
    assert sha(CKPT)==CONFIG['anew_checkpoint_sha256']
    assert sha(ROOT/'manifests/manifest.json')==CONFIG['manifest_sha256']
    full=manifest();groups={}
    for m,paths in json.loads((ROOT/'smoke_records.json').read_text()).items():
        original={r['key']:r for r in full['train'][m]};groups[m]=[]
        for key,v in paths.items():
            for kind in ['input','feature']:assert sha(v[kind+'_path'])==v[kind+'_sha256']
            groups[m].append({**original[key],'modality':m,'input_path':v['input_path'],'feature_path':v['feature_path']})
    native_batch,features=learning.native_batch,learning.features
    checks.native_batch=lambda rows:native_batch(rows,device=args.device)
    checks.features=lambda rows:features(rows,device=args.device)
    tracker=Tracker(stage,dict(optimizer_updates=0,formal_training=False,
        scope='Min/median/max cached real training examples; repeated fixtures for batch-size stress only'))
    tick=time.monotonic()
    report=dict(passed=False,optimizer_updates=0,device=args.device,seed=42,formal_training_started=False,cases=[])
    try:
        model,_=learning.load_official();model.eval().requires_grad_(False).to(args.device)
        frozen=list(model.parameters());fusion=Fusion(**CONFIG.get('fusion_kwargs', {})).to(args.device)
        epsilon=FusionEpsilon(model.diffusion.eps_net,fusion).to(args.device);model.diffusion.eps_net=epsilon
        initial=checks.state_hash(fusion.state_dict())
        native=checks.state_hash(dict(epsilon.base.named_parameters()))
        fixtures=[r for m in groups for r in sorted(groups[m],key=lambda r:r['nblocks'])]
        batch=[fixtures[i%len(fixtures)] for i in range(32)]
        cases=[('largest_'+m,[max(groups[m],key=lambda r:r['nblocks'])]) for m in groups]+[('effective_batch32',batch)]
        for label,rows in cases:
            model.train();model.autoencoder.eval();model.zero_grad(set_to_none=True)
            with checks.fixed_rng(42):losses,details,counts=backward_batch(model,epsilon,frozen,rows)
            grads=[p.grad for p in fusion.parameters() if p.requires_grad]
            assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
            norm=torch.nn.utils.clip_grad_norm_([p for p in fusion.parameters() if p.requires_grad],CONFIG['grad_clip'],error_if_nonfinite=True)
            value=dict(case=label,records=len(rows),max_blocks=max(r['nblocks'] for r in rows),losses=losses,
                counts=counts,gradient_norm=float(norm),gradient_finite=True)
            report['cases'].append(value);tracker.log(dict(case=label,**losses,gradient_norm=float(norm)))
        assert checks.state_hash(fusion.state_dict())==initial
        assert checks.state_hash(dict(epsilon.base.named_parameters()))==native
        report.update(passed=True,fusion_weights_unchanged=True,frozen_base_unchanged=True,seconds=time.monotonic()-tick,
            gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=='cuda' else None)
        atomic_json(dest,report);tracker.finish(report,[dest])
        print(json.dumps(report),flush=True)
    except BaseException as e:
        import traceback
        report.update(error=str(e),traceback=traceback.format_exc(),seconds=time.monotonic()-tick)
        atomic_json(dest,report);tracker.run.finish(exit_code=1);raise


if __name__=='__main__':main()
