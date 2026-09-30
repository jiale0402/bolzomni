"""Read-only checkpoint diagnostics. Does not retrain or replace test results."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from boltzgen.data import const
from poc_model import ROOT,load_model,move
from model import OmniCompletion,Observations,sample
from generate_panel import decode
from tracking import Tracker,file_hash

EXP=ROOT/'experiments/omni_joint'
LETTERS=[const.prot_token_to_letter[a] for a in const.canonical_tokens]

def decode_ids(x,feats):
    sequence,_,_=decode(x,feats)
    return torch.tensor([[LETTERS.index(a) if a in LETTERS else -1 for a in sequence]],device=x.device)

def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(1729)
    checkpoint=EXP/'runs/joint-seed-1729/selected.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    base=load_model();model=None;report={'checkpoint_step':state['step'],'examples':[]}
    rows=json.loads((EXP/'manifests/cache.json').read_text())
    rows=[r for r in rows if r['split']=='validation' and r['mode']=='design'][:4]
    tracker=Tracker(EXP/'runs/failure-diagnosis',dict(checkpoint_sha256=file_hash(checkpoint),
                    diagnostic_source_sha256=file_hash(__file__),validation_only=True),
                    'failure-diagnosis',job_type='diagnostic')
    for idx,row in enumerate(rows):
        p=move(torch.load(row['path'],weights_only=False,map_location='cpu'),'cuda')
        cache=p['cache'];f=cache['conditioning']['feats'];obs=Observations(**p['observations'])
        if model is None:
            model=OmniCompletion(base,cache['conditioning']['s_trunk'].shape[-1],cache['z_trunk'].shape[-1],
                      base.structure_module.score_model.a_norm.normalized_shape[0]).cuda().eval()
            model.load_state_dict(state['adapter'],strict=False)
        editable=~obs.identity_mask;target=p['target'];resolved=p['resolved'];labels=p['labels']
        target_ids=decode_ids(target,f)
        free=resolved*(~obs.coordinate_mask)
        center=(target*resolved[...,None]).sum(1,keepdim=True)/resolved.sum(1,keepdim=True)[...,None]
        item=dict(id=row['id'],tokens=int(labels.shape[1]),target_centroid=center.flatten().tolist(),
                  target_decode_accuracy=float((target_ids[editable]==labels[editable]).float().mean()),
                  target_decode_invalid=int((target_ids[editable]<0).sum()),
                  native_replay_eligible=cache['native_replay_eligible'],probes=[])
        with torch.no_grad():
            for sigma in (0.1,1.,10.,80.):
                torch.manual_seed(67);noise=torch.randn_like(target)*sigma
                for centered in (False,True):
                    clean=target-center if centered else target
                    anchors=obs.coordinates-center if centered else obs.coordinates
                    anchors=anchors.masked_fill(~obs.coordinate_mask[...,None],0)
                    shifted=Observations(obs.identity,obs.identity_mask,anchors,obs.coordinate_mask)
                    noisy=torch.where(obs.coordinate_mask[...,None],anchors,clean+noise)
                    pred,logits=model.denoise(cache,noisy,target.new_tensor([sigma]),obs.identity,shifted,target.new_ones(1))
                    gid=decode_ids(pred,f)
                    mse=float((((pred-clean)**2).sum(-1)*free).sum()/(3*free.sum()))
                    with model.native():
                        native,_=model.denoise(cache,noisy,target.new_tensor([sigma]),obs.identity,shifted,target.new_ones(1),native=True)
                    item['probes'].append(dict(sigma=sigma,centered=centered,mse=mse,
                         native_mse=float((((native-clean)**2).sum(-1)*free).sum()/(3*free.sum())),
                         head_accuracy=float((logits.argmax(-1)[editable]==labels[editable]).float().mean()),
                         geometry_accuracy=float((gid[editable]==labels[editable]).float().mean()),
                         head_geometry_agreement=float((gid[editable]==logits.argmax(-1)[editable]).float().mean())))
        if idx<2:
            # Log when identities become irreversible, and whether later geometry follows.
            original=model.denoise;trace=[];previous=obs.identity.clone();reveal_sigma=torch.full_like(labels,-1,dtype=torch.float)
            @torch.no_grad()
            def traced(*args,**kwargs):
                nonlocal previous
                pred,logits=original(*args,**kwargs)
                current=args[3];sigma=float(torch.as_tensor(args[2]).flatten()[0]);step=len(trace)
                newly=(previous==20)&(current!=20)
                if trace:reveal_sigma[newly]=trace[-1]['sigma']
                previous=current.clone()
                gid=decode_ids(pred,f)
                trace.append(dict(step=step,sigma=sigma,revealed=int((current[editable]!=20).sum()),
                      revealed_geometry_agreement=float((gid[editable&(current!=20)]==current[editable&(current!=20)]).float().mean()) if (editable&(current!=20)).any() else None,
                      head_geometry_agreement=float((gid[editable]==logits.argmax(-1)[editable]).float().mean())))
                return pred,logits
            model.denoise=traced
            try:result=sample(model,cache,obs,steps=500,seed=1729)
            finally:model.denoise=original
            gid=decode_ids(result['coordinates'],f);reveal_sigma[editable&(reveal_sigma<0)]=trace[-1]['sigma']
            final_agreement=gid==result['identity']
            item['trajectory']=trace
            item['sample']=dict(identity_geometry_agreement=float(final_agreement[editable].float().mean()),
                  geometry_invalid=int((gid[editable]<0).sum()),
                  last_head_geometry_agreement=float((result['identity_logits'].argmax(-1)[editable]==gid[editable]).float().mean()),
                  state_last_head_agreement=float((result['identity_logits'].argmax(-1)[editable]==result['identity'][editable]).float().mean()),
                  reveal_groups=[dict(low=low,high=high,count=int((editable&(reveal_sigma>=low)&(reveal_sigma<high)).sum()),
                    agreement=float(final_agreement[editable&(reveal_sigma>=low)&(reveal_sigma<high)].float().mean()) if (editable&(reveal_sigma>=low)&(reveal_sigma<high)).any() else None)
                    for low,high in ((0,1),(1,10),(10,100),(100,100000))])
        report['examples'].append(item)
        print('DIAGNOSED',json.dumps({k:v for k,v in item.items() if k!='trajectory'}),flush=True)
        tracker.log({'diagnostic/target_decode_accuracy':item['target_decode_accuracy'],
                     'diagnostic/target_centroid_norm':float(center.norm())},idx)
        (EXP/'results/failure_diagnosis.json').write_text(json.dumps(report,indent=2))
    import wandb
    artifact=wandb.Artifact('failure-diagnosis',type='diagnostic')
    artifact.add_file(str(EXP/'results/failure_diagnosis.json'));tracker.run.log_artifact(artifact);tracker.finish()

if __name__=='__main__':main()
