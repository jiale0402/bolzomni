"""Frozen native vs one frozen Omni checkpoint on a locked public benchmark subset."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
import biotite.structure as struc
from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.task.predict.data_from_yaml import Dataset,PredictionDataset,collate
from poc_model import ROOT as PROJECT,ASSETS,load_model,move,labels
from model import Observations,capture_context,sample
from full_model import FullOmni
from prompt import compile_prompt
from generate_panel import decode
from tracking import Tracker,file_hash
from tmtools import tm_align

ROOT=Path(__file__).resolve().parent
LETTERS=[const.prot_token_to_letter[a] for a in const.canonical_tokens]
AA3=dict(zip(LETTERS,const.canonical_tokens))

def write_backbone(path,bb,sequence):
    lines=[];serial=0
    for rid,(xyz,letter) in enumerate(zip(bb,sequence),1):
        for name,pos in zip(('N','CA','C','O'),xyz):
            serial+=1
            lines.append(f'ATOM  {serial:5d} {name:^4s} {AA3.get(letter,"UNK"):>3s} A{rid:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{1.:6.2f}{0.:6.2f}          {name[0]:>2s}  ')
    path.write_text('\n'.join(lines)+'\nTER\nEND\n')

def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    checkpoint=PROJECT/manifest['checkpoint'];ckpt_hash=file_hash(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==1000 and state['architecture']=='full-diffusion-no-lora'
    assert state['config']['source_sha256']['full_model.py']==file_hash(PROJECT/'experiments/omni_joint/full_model.py')
    outputs=ROOT/'results';outputs.mkdir(exist_ok=True)
    tracker=Tracker(outputs/'generation-tracking',dict(checkpoint_sha256=ckpt_hash,
        base_checkpoint_sha256=state['config']['base_checkpoint_sha256'],
        manifest_sha256=file_hash(ROOT/'manifest.json'),source_sha256=file_hash(__file__),
        sampling_steps=500,recycles=3,folding_targets=10,design_targets=6,candidates_per_design=2,
        methods=['native','omni'],training_performed=False,
        design_evaluator='Official MotifBench sequence-redesign/refolding success, no identity-agreement gate'),
        'standard-benchmark-subset-generation',job_type='evaluation')
    ok=False;started=time.monotonic()
    try:
        teacher=load_model();student=load_model();model=None
        canonicals=load_canonicals(str(ASSETS/'mols.zip'))
        allrows=[]
        for row in manifest['folding']:
            path=ROOT/'prompts'/(row['id']+'.yaml')
            path.write_text(yaml.safe_dump({'entities':[{'protein':{'id':'A','sequence':row['sequence']}}]},sort_keys=False))
            allrows.append(dict(row,mode='folding',yaml=str(path),seed=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)%(2**31)))
        allrows += [dict(row,mode='design') for row in manifest['design']]
        for index,row in enumerate(allrows):
            if all((outputs/f'{row["id"]}-{method}.json').exists() for method in ('native','omni')):continue
            seed=row['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            dataset=PredictionDataset(Dataset(row['yaml'],Tokenizer(atomize_modified_residues=False),Featurizer()),
                canonicals,str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False)
            rng=np.random.default_rng;np.random.default_rng=lambda value=None:rng(seed if value is None else value)
            try: f=move(collate([dataset[0]]),'cuda')
            finally:np.random.default_rng=rng
            prompt=compile_prompt(teacher,f,canonical_frame=True)
            cache=capture_context(teacher,prompt)
            if model is None:
                model=FullOmni(student,teacher,cache).cuda().eval()
                assert set(state['adapter'])=={n for n,p in model.named_parameters() if p.requires_grad}
                model.load_state_dict(state['adapter'],strict=False)
                del state
            n=prompt['res_type'].shape[1];assert n==row['length']
            identity_mask=~prompt['design_mask'].bool()
            visible=prompt['token_distance_mask'].bool().any(-1)
            atom_visible=torch.bmm(prompt['atom_to_token'].float(),visible.float()[...,None]).squeeze(-1).bool()
            # compile_prompt exposes backbone geometry only. Sidechains are not
            # coordinate observations, even when residue identity is specified.
            hard=atom_visible & prompt['backbone_mask'].bool() & prompt['atom_pad_mask'].bool()
            ids=labels(f);ids[~identity_mask]=20
            coordinates=prompt['coords'].reshape(1,-1,3).masked_fill(~hard[...,None],0)
            obs=Observations(ids,identity_mask,coordinates,hard)
            if row['mode']=='folding':
                assert not hard.any() and not prompt['coords'].any() and identity_mask.all()
            bbmap=prompt['token_to_bb4_atoms'].float().reshape(1,n*4,-1)
            for method in ('native','omni'):
                path=outputs/f'{row["id"]}-{method}.json'
                if path.exists():continue
                begin=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    if method=='native':
                        x=teacher.structure_module.sample(**cache['conditioning'],atom_mask=prompt['atom_pad_mask'].float(),
                            multiplicity=1,num_sampling_steps=500,step_scale=2.,noise_scale=.88)['sample_atom_coords']
                        result=None
                    else:
                        result=sample(model,cache,obs,steps=500,seed=seed);x=result['coordinates']
                geometric,_,_=decode(x,prompt)
                sequence=geometric if method=='native' else ''.join(LETTERS[i] for i in result['identity'][0].tolist())
                if row['mode']=='folding':sequence=row['sequence']
                bb=torch.bmm(bbmap,x)[0].reshape(n,4,3).cpu().numpy()
                record=dict(id=row['id'],mode=row['mode'],method=method,seed=seed,sequence=sequence,
                    geometric_sequence=geometric,seconds=time.monotonic()-begin,
                    finite_geometry=bool(np.isfinite(bb).all()),checkpoint_sha256=ckpt_hash if method=='omni' else None)
                if row['mode']=='folding':
                    reference=row['resolved_ca'];indices=sorted(map(int,reference))
                    true=np.asarray([reference[str(i)] for i in indices],dtype=float)
                    pred=bb[np.asarray(indices)-1,1].astype(float)
                    ref=struc.AtomArray(len(indices));ref.coord=true;ref.res_id=np.asarray(indices)
                    ref.atom_name[:]='CA';ref.element[:]='C';ref.chain_id[:]='A'
                    ref.res_name[:]=[AA3[sequence[i-1]] for i in indices]
                    subject=ref.copy();subject.coord=pred
                    seq=''.join(sequence[i-1] for i in indices)
                    record.update(ca_lddt=float(struc.lddt(ref,subject)) if record['finite_geometry'] else 0.,
                        tm_score=float(tm_align(pred,true,seq,seq).tm_norm_chain2) if record['finite_geometry'] else 0.,
                        resolved_residues=len(indices))
                    dest=outputs/'folding'/method;dest.mkdir(parents=True,exist_ok=True)
                    write_backbone(dest/(row['id']+'.pdb'),bb,sequence)
                else:
                    editable=(~identity_mask)[0].cpu().numpy()
                    agreement=np.asarray([a==b and a in LETTERS for a,b in zip(sequence,geometric)])
                    record.update(task=row['case'],group=row['group'],candidate=row['candidate'],
                        identity_geometry_agreement=float(agreement[editable].mean()),
                        geometry_unknown=geometric.count('X'))
                    dest=outputs/'scaffolds'/method/row['case'];dest.mkdir(parents=True,exist_ok=True)
                    write_backbone(dest/(row['id']+'.pdb'),bb,sequence)
                    (dest/'scaffold_info.csv').write_text('sample_num,motif_placements\n'+''.join(f'{i},{row["motif_placements"]}\n' for i in range(2)))
                np.savez_compressed(outputs/f'{row["id"]}-{method}.npz',backbone=bb)
                path.write_text(json.dumps(record,indent=2))
                tracker.log({method+'/'+row['mode']+'/'+k:v for k,v in record.items() if isinstance(v,(float,int))},index)
                print('BENCHMARK_PREDICTED',row['id'],method,json.dumps({k:record[k] for k in ('ca_lddt','tm_score','identity_geometry_agreement') if k in record}),flush=True)
        import wandb
        artifact=wandb.Artifact('standard-benchmark-subset-predictions',type='evaluation')
        artifact.add_file(str(ROOT/'manifest.json'))
        artifact.add_dir(str(outputs/'folding'),name='folding')
        artifact.add_dir(str(outputs/'scaffolds'),name='scaffolds')
        tracker.run.log_artifact(artifact)
        tracker.log({'compute/GPU_hours':(time.monotonic()-started)/3600,'compute/peak_memory_GB':torch.cuda.max_memory_allocated()/1e9},len(allrows))
        ok=True
    finally:tracker.finish(ok)

if __name__=='__main__':main()
