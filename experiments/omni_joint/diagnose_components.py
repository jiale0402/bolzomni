"""Validation-only interventions. Original checkpoint and scored outputs are untouched."""
import inspect
import json
import torch
from poc_model import ROOT,load_model,move
import model as original_model
from model import OmniCompletion,Observations,sample
from generate_panel import decode
from tracking import Tracker,file_hash

EXP=ROOT/'experiments/omni_joint'

def late_sampler():
    source=inspect.getsource(sample)
    old="    state = observations.identity.clone()"
    source=source.replace(old,old+"\n    late_steps = sum(float(sigmas[k])*(1+float(gammas[k+1]))<=1.0 for k in range(steps))\n    assert late_steps>0")
    source=source.replace('chemical_t = x.new_tensor([(steps-k)/steps])',
                          'chemical_t = x.new_tensor([min(1.0,(steps-k)/late_steps)])')
    source=source.replace('        state = torch.where(reveal,proposed,state)',
                          '        reveal &= t_hat <= 1.0\n        state = torch.where(reveal,proposed,state)')
    namespace=dict(vars(original_model));exec(compile(source,'<diagnostic-late-sampler>','exec'),namespace)
    return namespace['sample'],source

def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(1729)
    late,source=late_sampler();(EXP/'results/diagnostic_late_sampler.py').write_text(source)
    checkpoint=EXP/'runs/joint-seed-1729/selected.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    rows=json.loads((EXP/'manifests/cache.json').read_text())
    rows=[r for r in rows if r['split']=='validation' and r['mode']=='design'][:2]
    base=load_model();model=None;records=[]
    tracker=Tracker(EXP/'runs/component-controls',dict(checkpoint_sha256=file_hash(checkpoint),
               validation_only=True,source_sha256=file_hash(__file__),late_threshold_sigma=1.),
               'component-failure-controls',job_type='diagnostic')
    for row in rows:
        p=move(torch.load(row['path'],map_location='cpu',weights_only=False),'cuda')
        cache=p['cache'];f=cache['conditioning']['feats'];obs=Observations(**p['observations'])
        if model is None:
            model=OmniCompletion(base,cache['conditioning']['s_trunk'].shape[-1],cache['z_trunk'].shape[-1],
                      base.structure_module.score_model.a_norm.normalized_shape[0]).cuda().eval()
            zero={k:v.clone() for k,v in model.trainable_state().items()}
        for variant in ('no_lora','no_feedback','all_masked'):
            model.load_state_dict(zero if variant=='zero_hard' else state['adapter'],strict=False)
            torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
            if variant=='native':
                with torch.no_grad(),model.native():
                    x=base.structure_module.sample(**cache['conditioning'],atom_mask=f['atom_pad_mask'].float(),
                          multiplicity=1,num_sampling_steps=500,step_scale=2.,noise_scale=.88)['sample_atom_coords']
                result=None
            else:
                if variant=='no_feedback':
                    with torch.no_grad():
                        model.adapter.token_out.weight.zero_();model.adapter.token_out.bias.zero_()
                        model.adapter.pair_out.weight.zero_()
                if variant=='no_lora':
                    with model.native(): result=sample(model,cache,obs,steps=500,seed=1729)
                elif variant=='all_masked':
                    original=model.denoise
                    def masked(cache,x,sigma,state,observation,t,**kwargs):
                        return original(cache,x,sigma,obs.identity,observation,torch.ones_like(t),**kwargs)
                    model.denoise=masked
                    try:result=sample(model,cache,obs,steps=500,seed=1729)
                    finally:model.denoise=original
                else: result=sample(model,cache,obs,steps=500,seed=1729)
                x=result['coordinates']
            sequence,ca,_=decode(x,f);mask=(~obs.identity_mask)[0]
            editable=[i for i in range(len(sequence)) if bool(mask[i])]
            lengths=(ca[1:]-ca[:-1]).norm(dim=-1)
            record=dict(id=row['id'],variant=variant,editable_residues=len(editable),
                        geometric_invalid_residues=sum(sequence[i]=='X' for i in editable),
                        abnormal_CA_bonds=int(((lengths<3.4)|(lengths>4.2)).sum()))
            if result is not None:
                from boltzgen.data import const
                letters=[const.prot_token_to_letter[a] for a in const.canonical_tokens]
                proposed=''.join(letters[i] for i in result['identity'][0].tolist())
                record['identity_geometry_agreement']=sum(proposed[i]==sequence[i] for i in editable)/len(editable)
            records.append(record);print('CONTROL',json.dumps(record),flush=True)
            tracker.log({'diagnostic/'+variant+'/'+k:v for k,v in record.items() if isinstance(v,(int,float))},len(records))
            (EXP/'results/component_controls.json').write_text(json.dumps(records,indent=2))
    import wandb
    artifact=wandb.Artifact('component-failure-controls',type='diagnostic')
    for name in ('component_controls.json','diagnostic_late_sampler.py'):artifact.add_file(str(EXP/'results'/name))
    tracker.run.log_artifact(artifact);tracker.finish()

if __name__=='__main__':main()
