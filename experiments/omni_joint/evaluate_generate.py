"""Paired held-out generation using one locked checkpoint, with raw failures retained."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from boltzgen.data import const
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals
from boltzgen.task.predict.data_from_yaml import Dataset,PredictionDataset,collate
from poc_model import ROOT,ASSETS,load_model,move,labels
from generate_panel import decode
from prompt import compile_prompt
from model import OmniCompletion,Observations,capture_context,sample,endpoint_identity_consistency
from metrics import ca_lddt
from summarize_qualification import aligned_rmsd
from tracking import Tracker,file_hash

EXP=ROOT/'experiments/omni_joint'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--steps',type=int,default=500)
    ap.add_argument('--validation-only',action='store_true')
    args=ap.parse_args()
    assert args.validation_only or (args.limit==0 and args.steps==500)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    checkpoint=EXP/'runs/joint-seed-1729/selected.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    output=EXP/'results'/('paired-validation' if args.validation_only else 'paired-test')
    output.mkdir(parents=True,exist_ok=True)
    root_data=json.loads((EXP/'manifests/data_filtered.json').read_text())
    rows=root_data['validation' if args.validation_only else 'test']
    prompts=[]
    for index,row in enumerate(rows[:args.limit or None]):
        path=output/(row['id']+'-folding.yaml')
        spec=dict(path=str(Path(row['directory'])/'native.pdb'),include=[{'chain':{'id':'A'}}],
                  structure_groups=[{'group':{'id':'A','visibility':0}}])
        path.write_text(yaml.safe_dump({'entities':[{'file':spec}]},sort_keys=False))
        prompts.append(dict(id=row['id'],mode='folding',yaml=str(path),seed=100000+index,
                            design_length=row['length']))
        if args.validation_only:
            n=row['length'];start=n//2-8;motif=list(range(start,start+16))
            spec['design']=[{'chain':{'id':'A','res_index':','.join(str(i+1) for i in range(n) if i not in motif)}}]
            spec['structure_groups'].append({'group':{'id':'A','visibility':1,'res_index':','.join(str(i+1) for i in motif)}})
            path=output/(row['id']+'-design.yaml')
            path.write_text(yaml.safe_dump({'entities':[{'file':spec}]},sort_keys=False))
            prompts.append(dict(id=row['id']+'-design',mode='design',yaml=str(path),seed=100000+index,
                                design_length=n,task=row['id'],candidate=0,
                                motif_mapping=[dict(output_residue=i+1) for i in motif]))
    if not args.validation_only:
        for row in json.loads((EXP/'manifests/motifs.json').read_text()):
            prompts.append(dict(row,id=row['task']+'-c'+str(row['candidate']).zfill(2),mode='design'))
    base=load_model();model=None;canonicals=load_canonicals(str(ASSETS/'mols.zip'))
    tracker=Tracker(output/'tracking',dict(checkpoint_sha256=file_hash(checkpoint),
                generation_source_sha256=file_hash(__file__),
                checkpoint_step=state['step'],sampling_steps=args.steps,formal_test=not args.validation_only,
                data_manifest_sha256=file_hash(EXP/'manifests/data_filtered.json'),
                motif_manifest_sha256=file_hash(EXP/'manifests/motifs.json')),
                'paired-held-out-generation' if not args.validation_only else 'paired-validation-generation',job_type='evaluation')
    successful=False
    try:
        for index,row in enumerate(prompts):
            if all((output/(row['id']+'-'+method+'.json')).exists() for method in ('native','omni')):continue
            seed=row['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            ds=PredictionDataset(Dataset(row['yaml'],Tokenizer(atomize_modified_residues=False),Featurizer()),
                  canonicals,str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False)
            rng=np.random.default_rng;np.random.default_rng=lambda value=None:rng(seed if value is None else value)
            try:f=move(collate([ds[0]]),'cuda')
            finally:np.random.default_rng=rng
            truth=f['coords'].reshape(1,-1,3).clone();lab=labels(f)
            n=int(f['res_type'].shape[1]);length=row['design_length']
            bb_map=f['token_to_bb4_atoms'].float().reshape(1,n*4,-1)
            truth_bb=torch.bmm(bb_map,truth)[0].reshape(n,4,3).cpu().numpy()[:length]
            prompt=compile_prompt(base,f,canonical_frame=True)
            if row['mode']=='folding':
                assert not prompt['coords'].any() and not prompt['center_coords'].any() and not prompt['token_distance_mask'].any()
            cache=capture_context(base,prompt)
            if model is None:
                model=OmniCompletion(base,cache['conditioning']['s_trunk'].shape[-1],cache['z_trunk'].shape[-1],
                                     base.structure_module.score_model.a_norm.normalized_shape[0]).cuda().eval()
                expected={name for name,p in model.named_parameters() if p.requires_grad}
                assert set(state['adapter'])==expected
                model.load_state_dict(state['adapter'],strict=False)
            identity_mask=~prompt['design_mask'].bool()
            visible=prompt['token_distance_mask'].bool().any(-1)
            atom_visible=torch.bmm(prompt['atom_to_token'].float(),visible.float()[...,None]).squeeze(-1).bool()
            hard=atom_visible & prompt['backbone_mask'].bool() & prompt['atom_pad_mask'].bool()
            identities=lab.clone();identities[~identity_mask]=20
            obs=Observations(identities,identity_mask,prompt['coords'].reshape_as(truth).masked_fill(~hard[...,None],0),hard)
            for method in ('native','omni'):
                stem=row['id']+'-'+method;destination=output/(stem+'.json')
                if destination.exists():continue
                started=time.monotonic()
                if method=='native':
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    with torch.no_grad(),model.native():
                        x=base.structure_module.sample(**cache['conditioning'],atom_mask=prompt['atom_pad_mask'].float(),
                              multiplicity=1,num_sampling_steps=args.steps,step_scale=2.,noise_scale=.88)['sample_atom_coords']
                    result=None
                else:
                    result=sample(model,cache,obs,steps=args.steps,seed=seed);x=result['coordinates']
                geometric,_,_=decode(x,prompt)
                sequence=geometric
                consistent=True
                if result is not None:
                    letters=[const.prot_token_to_letter[aa] for aa in const.canonical_tokens]
                    sequence=''.join(letters[i] if 0<=i<20 else 'X' for i in result['identity'][0].tolist())
                    geometric_ids=torch.tensor([[letters.index(a) if a in letters else -1 for a in geometric]],device='cuda')
                    consistent,_=endpoint_identity_consistency(result['identity'],geometric_ids,~identity_mask)
                backbone=torch.bmm(bb_map,x)[0].reshape(n,4,3).cpu().numpy()[:length]
                sequence=sequence[:length];geometric=geometric[:length]
                record=dict(id=row['id'],method=method,mode=row['mode'],seed=seed,sequence=sequence,
                            geometric_sequence=geometric,identity_geometry_consistent=consistent,
                            sequence_valid='X' not in sequence,finite_geometry=bool(np.isfinite(backbone).all()),
                            seconds=time.monotonic()-started,checkpoint_sha256=file_hash(checkpoint) if method=='omni' else None)
                if row['mode']=='folding':
                    record['ca_lddt']=ca_lddt(backbone[:,1],truth_bb[:,1])
                    record['ca_rmsd']=None;record['tm_score']=0.
                    if record['finite_geometry']:
                        record['ca_rmsd']=aligned_rmsd(backbone[:,1],truth_bb[:,1])
                        from tmtools import tm_align
                        record['tm_score']=float(tm_align(backbone[:,1].astype(float),truth_bb[:,1].astype(float),sequence,sequence).tm_norm_chain2)
                    motif=[]
                else:
                    motif=[r['output_residue']-1 for r in row['motif_mapping']]
                    record.update(task=row['task'],candidate=row['candidate'],motif=motif,
                                  raw_motif_rmsd=aligned_rmsd(backbone[motif].reshape(-1,3),truth_bb[motif].reshape(-1,3)) if record['finite_geometry'] else None)
                np.savez_compressed(output/(stem+'.npz'),backbone=backbone,native_backbone=truth_bb,motif=motif)
                destination.write_text(json.dumps(record,indent=2))
                tracker.log({method+'/'+row['mode']+'/'+k:v for k,v in record.items() if isinstance(v,(int,float))},index)
                print('EVALUATED',stem,record.get('ca_lddt'),record['identity_geometry_consistent'],flush=True)
        fasta=[]
        for path in sorted(output.glob('*.json')):
            record=json.loads(path.read_text())
            if record.get('mode')=='design' and record['sequence_valid'] and record['finite_geometry'] and record['identity_geometry_consistent']:
                fasta.append('>'+path.stem+'\n'+record['sequence'])
        (output/'designs.fasta').write_text('\n'.join(fasta)+'\n' if fasta else '')
        successful=True
    finally:tracker.finish(successful)

if __name__=='__main__':main()
