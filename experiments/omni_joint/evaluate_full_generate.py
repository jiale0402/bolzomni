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
from model import Observations,capture_context,sample,endpoint_identity_consistency
from full_model import FullOmni
import shutil
from metrics import ca_lddt
from summarize_qualification import aligned_rmsd
from tracking import Tracker,file_hash

EXP=ROOT/'experiments/omni_joint'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--steps',type=int,default=500)
    ap.add_argument('--validation-only',action='store_true')
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args()
    assert args.validation_only or (args.limit==0 and args.steps==500)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    checkpoint=EXP/'runs/full-poc-seed-1729/final.pt'
    checkpoint_hash=file_hash(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert not args.validation_only and args.limit==0 and args.steps==500
    output=EXP/'results/paired-full-poc'
    output.mkdir(parents=True,exist_ok=True)
    assert 0<=args.shard<args.shards
    prompt_dir=output/f'prompts-{args.shard}'
    prompt_dir.mkdir(exist_ok=True)
    root_data=json.loads((EXP/'manifests/data_filtered.json').read_text())
    manifest=json.loads((EXP/'runs/full-poc-seed-1729/evaluation_manifest.json').read_text())
    rows=manifest['folding']
    original_indices={r['id']:i for i,r in enumerate(root_data['test'])}
    prompts=[]
    for index,row in enumerate(rows[:args.limit or None]):
        path=prompt_dir/(row['id']+'-folding.yaml')
        spec=dict(path=str(Path(row['directory'])/'native.pdb'),include=[{'chain':{'id':'A'}}],
                  structure_groups=[{'group':{'id':'A','visibility':0}}])
        path.write_text(yaml.safe_dump({'entities':[{'file':spec}]},sort_keys=False))
        prompts.append(dict(id=row['id'],mode='folding',yaml=str(path),seed=100000+original_indices[row['id']],
                            design_length=row['length']))
        if args.validation_only:
            n=row['length'];start=n//2-8;motif=list(range(start,start+16))
            spec['design']=[{'chain':{'id':'A','res_index':','.join(str(i+1) for i in range(n) if i not in motif)}}]
            spec['structure_groups'].append({'group':{'id':'A','visibility':1,'res_index':','.join(str(i+1) for i in motif)}})
            path=output/(row['id']+'-design.yaml')
            path.write_text(yaml.safe_dump({'entities':[{'file':spec}]},sort_keys=False))
            prompts.append(dict(id=row['id']+'-design',mode='design',yaml=str(path),seed=100000+original_indices[row['id']],
                                design_length=n,task=row['id'],candidate=0,
                                motif_mapping=[dict(output_residue=i+1) for i in motif]))
    if not args.validation_only:
        for row in manifest['design']:
            prompts.append(dict(row,id=row['task']+'-c'+str(row['candidate']).zfill(2),mode='design'))
    prompts=[row for i,row in enumerate(prompts) if i%args.shards==args.shard]
    old_lock=json.loads((EXP/'LOCK.json').read_text())
    assert old_lock['base_checkpoint_sha256']==state['config']['base_checkpoint_sha256']
    for source in ('prompt.py','model.py','../../scripts/generate_panel.py','../../scripts/poc_model.py'):
        assert file_hash(EXP/source)==old_lock['source_sha256'][source], ('Native reuse source mismatch',source)
    old=EXP/'results/paired-test'
    # The original evaluator was finalized after the old training lock. Its own
    # completed W&B generation run, not that earlier lock, pins its actual source.
    provenance=json.loads((old/'native_generation_provenance.json').read_text())['config']
    assert provenance['generation_source_sha256']==file_hash(EXP/'evaluate_generate.py')
    assert provenance['data_manifest_sha256']==file_hash(EXP/'manifests/data_filtered.json')
    assert provenance['motif_manifest_sha256']==file_hash(EXP/'manifests/motifs.json')
    assert provenance['sampling_steps']==500
    (output/'af2').mkdir(exist_ok=True)
    reused=[]
    for row in prompts:
        stem=row['id']+'-native'
        source=old/(stem+'.json')
        if not source.exists(): continue
        record=json.loads(source.read_text())
        assert record['seed']==row['seed'] and record['mode']==row['mode'] and record['method']=='native'
        assert len(record['sequence'])==row['design_length']
        for suffix in ('.json','.npz'): shutil.copy2(old/(stem+suffix),output/(stem+suffix))
        for f in (old/'af2').glob(stem+'_*'): shutil.copy2(f,output/'af2'/f.name)
        reused.append(stem)
    (output/f'native_reuse-{args.shard}.json').write_text(json.dumps(dict(ids=reused,original_sampling_steps=500,original_source_lock_sha256=file_hash(EXP/'LOCK.json')),indent=2))
    base=load_model();student=load_model();model=None;canonicals=load_canonicals(str(ASSETS/'mols.zip'))
    tracker=Tracker(output/f'tracking-{args.shard}',dict(checkpoint_sha256=checkpoint_hash,
                generation_source_sha256=file_hash(__file__),
                shard=args.shard,shards=args.shards,
                checkpoint_step=state['step'],sampling_steps=args.steps,formal_test=False,exploratory=True,
                data_manifest_sha256=file_hash(EXP/'manifests/data_filtered.json'),
                motif_manifest_sha256=file_hash(EXP/'manifests/motifs.json')),
                f'full-poc-paired-generation-{args.shard}',job_type='evaluation')
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
                model=FullOmni(student,base,cache).cuda().eval()
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
                            seconds=time.monotonic()-started,checkpoint_sha256=checkpoint_hash if method=='omni' else None)
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
        (output/f'designs-shard-{args.shard}.fasta').write_text('\n'.join(fasta)+'\n' if fasta else '')
        successful=True
    finally:tracker.finish(successful)

if __name__=='__main__':main()
