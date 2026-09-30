"""Same locked panel/metrics; fresh native and final Omni on the same allocated GPU.

Derived from native_condition_poc/generate.py. No control training or selection.
"""
import argparse
import os
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
import biotite.structure as struc
from tmtools import tm_align
from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.task.predict.data_from_yaml import Dataset,PredictionDataset,collate
from boltzgen.model.loss.confidence import lddt_dist
from poc_model import ROOT as PROJECT,ASSETS,load_model,move,labels
from model import Observations,capture_context
from noise_matched_flow import FullOmni,sample
from prompt import compile_prompt
from generate_panel import decode
from tracking import Tracker,file_hash
from raw_joint import chemical_validity

ROOT=Path(__file__).resolve().parent
LETTERS=[const.prot_token_to_letter[a] for a in const.canonical_tokens]
AA3=dict(zip(LETTERS,const.canonical_tokens))


def write_backbone(path,bb,sequence):
    lines=[]
    for rid,(xyz,letter) in enumerate(zip(bb,sequence),1):
        for name,pos in zip(('N','CA','C','O'),xyz):
            lines.append(f'ATOM  {len(lines)+1:5d} {name:^4s} {AA3.get(letter,"UNK"):>3s} A{rid:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{1.:6.2f}{0.:6.2f}          {name[0]:>2s}  ')
    path.write_text('\n'.join(lines)+'\nTER\nEND\n')


def score_folding(x,bb,f,row):
    seq=row['sequence'];indices=sorted(map(int,row['resolved_ca']))
    true=np.asarray([row['resolved_ca'][str(i)] for i in indices],dtype=float)
    pred=bb[np.asarray(indices)-1,1].astype(float)
    ref=struc.AtomArray(len(indices));ref.coord=true;ref.res_id=np.asarray(indices)
    ref.atom_name[:]='CA';ref.element[:]='C';ref.chain_id[:]='A';ref.res_name[:]=[AA3[seq[i-1]] for i in indices]
    subject=ref.copy();subject.coord=pred;sequence=''.join(seq[i-1] for i in indices)
    scores=dict(ca_lddt=float(struc.lddt(ref,subject)),tm_score=float(tm_align(pred,true,sequence,sequence).tm_norm_chain2))
    # Additional native all-atom distance score with explicit atom-name matching.
    reference={}
    for line in (Path(row['directory'])/'native.pdb').read_text().splitlines():
        if line.startswith('ATOM'):
            reference[(int(line[22:26]),line[12:16].strip())]=[float(line[30:38]),float(line[38:46]),float(line[46:54])]
    names=f['ref_atom_name_chars'][0].argmax(-1).cpu().numpy()
    residues=f['atom_to_token'][0].long().argmax(-1).cpu().numpy()+1
    used=[];coords=[]
    for i in range(len(names)):
        if not bool(f['atom_pad_mask'][0,i]):continue
        name=''.join(chr(int(a)+32) for a in names[i] if a!=0)
        if (int(residues[i]),name) in reference:
            used.append(i);coords.append(reference[(int(residues[i]),name)])
    assert len(used)>=len(indices)*4*.7
    true_t=x.new_tensor(coords)[None];pred_t=x[:,used]
    mask=(1-torch.eye(len(used),device=x.device))[None]
    score,_=lddt_dist(torch.cdist(pred_t,pred_t),torch.cdist(true_t,true_t),mask)
    scores.update(all_atom_lddt=float(score),all_atom_matched=len(used),symmetry_correction=False)
    return scores


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorized-continuation',action='store_true')
    args=parser.parse_args()
    assert args.authorized_continuation and os.environ.get('SLURM_JOB_ID'), 'Requires separately approved continuation budget'
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    manifest=json.loads((ROOT/'benchmark_manifest.json').read_text())
    checkpoint=PROJECT/manifest['checkpoint'];ckpt_hash=file_hash(checkpoint)
    native_hash=file_hash(ASSETS/'boltzgen1_diverse.ckpt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==2000 and state['architecture']=='noise-matched-full-diffusion-no-lora'
    output=ROOT/'results';output.mkdir(exist_ok=True);(ROOT/'prompts').mkdir(exist_ok=True)
    provenance=dict(checkpoint_sha256=ckpt_hash,native_checkpoint_sha256=native_hash,
        manifest_sha256=file_hash(ROOT/'benchmark_manifest.json'),
        source_sha256=file_hash(__file__),slurm_job_id=os.environ['SLURM_JOB_ID'],
        GPU=torch.cuda.get_device_name(),node=os.environ.get('SLURMD_NODENAME'))
    provenance_path=output/'generation_provenance.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text())==provenance, 'Do not mix generations across checkpoints, sources or jobs'
    else:provenance_path.write_text(json.dumps(provenance,indent=2))
    tracker=Tracker(output/'generation-tracking',dict(checkpoint_sha256=ckpt_hash,manifest_sha256=file_hash(ROOT/'benchmark_manifest.json'),
        source_sha256=file_hash(__file__),GPU=torch.cuda.get_device_name(),native_checkpoint_sha256=native_hash,chemical_schedule='coordinate_noise_cdf',folding_samples=5,design_samples=4,sampling_steps=500,training_performed=False),
        'noise-matched-paired-generation',job_type='evaluation')
    ok=False;started=time.monotonic()
    try:
        teacher=load_model();student=load_model();model=None;canonicals=load_canonicals(str(ASSETS/'mols.zip'))
        jobs=[]
        for row in manifest['folding']:
            path=ROOT/'prompts'/(row['id']+'.yaml')
            path.write_text(yaml.safe_dump({'entities':[{'protein':{'id':'A','sequence':row['sequence']}}]},sort_keys=False))
            for candidate in range(5):
                seed=int(hashlib.sha256(f'official:{row["id"]}:{candidate}'.encode()).hexdigest()[:8],16)%(2**31)
                jobs.append(dict(row,target=row['id'],id=f'{row["id"]}_{candidate}',candidate=candidate,mode='folding',yaml=str(path),seed=seed))
        jobs += [dict(r,mode='design') for r in manifest['design']]
        for index,row in enumerate(jobs):
            if all((output/f'{row["id"]}-{method}.json').exists() for method in ('native','omni')):continue
            seed=row['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            dataset=PredictionDataset(Dataset(row['yaml'],Tokenizer(atomize_modified_residues=False),Featurizer()),canonicals,str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False)
            rng=np.random.default_rng;np.random.default_rng=lambda value=None:rng(seed if value is None else value)
            try:f=move(collate([dataset[0]]),'cuda')
            finally:np.random.default_rng=rng
            prompt=compile_prompt(teacher,f,canonical_frame=True);cache=capture_context(teacher,prompt)
            if model is None:
                model=FullOmni(student,teacher,cache).cuda().eval()
                assert set(state['adapter'])=={n for n,p in model.named_parameters() if p.requires_grad}
                model.load_state_dict(state['adapter'],strict=False);del state
            n=prompt['res_type'].shape[1];assert n==row['length']
            identity_mask=~prompt['design_mask'].bool();visible=prompt['token_distance_mask'].bool().any(-1)
            atom_visible=torch.bmm(prompt['atom_to_token'].float(),visible.float()[...,None]).squeeze(-1).bool()
            relative=atom_visible & prompt['backbone_mask'].bool() & prompt['atom_pad_mask'].bool()
            hard=torch.zeros_like(relative)  # Relative groups stay in native features.
            ids=labels(f);ids[~identity_mask]=20
            obs=Observations(ids,identity_mask,prompt['coords'].reshape(1,-1,3).masked_fill(~hard[...,None],0),hard)
            if row['mode']=='folding':assert not hard.any() and not prompt['coords'].any() and identity_mask.all()
            else:
                observed=set((identity_mask[0].nonzero().flatten()+1).tolist())
                assert observed==set(row['motif_indices'])
            bbmap=prompt['token_to_bb4_atoms'].float().reshape(1,n*4,-1)
            for method in ('native','omni'):
                path=output/f'{row["id"]}-{method}.json'
                if path.exists():continue
                begin=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    if method=='native':
                        x=teacher.structure_module.sample(**cache['conditioning'],atom_mask=prompt['atom_pad_mask'].float(),multiplicity=1,num_sampling_steps=500,step_scale=2.,noise_scale=.88)['sample_atom_coords'];result=None
                    else:result=sample(model,cache,obs,steps=500,seed=seed);x=result['coordinates']
                geometric,_,_=decode(x,prompt)
                sequence=geometric if result is None else ''.join(LETTERS[i] for i in result['identity'][0].tolist())
                if row['mode']=='folding':sequence=row['sequence']
                bb=torch.bmm(bbmap,x)[0].reshape(n,4,3).cpu().numpy()
                gaps=np.linalg.norm(np.diff(bb[:,1],axis=0),axis=-1)
                record=dict(max_ca_distance=float(gaps.max()),abnormal_ca_distances=int(((gaps<3.4)|(gaps>4.2)).sum()),id=row['id'],target=row.get('target',row.get('case')),mode=row['mode'],method=method,seed=seed,candidate=row['candidate'],
                    sequence=sequence,geometric_sequence=geometric,seconds=time.monotonic()-begin,finite_geometry=bool(np.isfinite(bb).all()),checkpoint_sha256=ckpt_hash if method=='omni' else native_hash)
                if row['mode']=='folding':
                    record.update(score_folding(x,bb,f,row) if record['finite_geometry'] else dict(ca_lddt=0.,tm_score=0.,all_atom_lddt=0.))
                    dest=output/'folding'/method;dest.mkdir(parents=True,exist_ok=True)
                else:
                    editable=(~identity_mask)[0].cpu().numpy();agreement=np.asarray([a==b and a in LETTERS for a,b in zip(sequence,geometric)])
                    record.update(task=row['case'],group=row['group'],identity_geometry_agreement=float(agreement[editable].mean()),geometry_unknown=geometric.count('X'))
                    # Motif identity is a supplied observation in both arms.
                    native_letters=''.join(LETTERS[i] for i in labels(f)[0].tolist())
                    expected_observed={i+1:letter for i,letter in enumerate(native_letters) if bool(identity_mask[0,i])}
                    record['raw_validity']=chemical_validity(sequence,geometric,expected_observed,
                        finite_geometry=record['finite_geometry'])
                    sequence=''.join(native_letters[i] if bool(identity_mask[0,i]) else letter for i,letter in enumerate(sequence))
                    # Keep the exact model sequence for raw refolding. The
                    # existing scaffold writer supplies motif identities to MPNN.
                    record['scaffold_writer_sequence']=sequence
                    dest=output/'scaffolds'/method/row['case'];dest.mkdir(parents=True,exist_ok=True)
                    case=next(c for c in manifest['design_cases'] if c['case']==row['case'])
                    with (dest/'scaffold_info.csv').open('w',newline='') as handle:
                        writer=csv.DictWriter(handle,fieldnames=['sample_num','motif_placements']);writer.writeheader();writer.writerows(case['scaffold_info'])
                write_backbone(dest/(row['id']+'.pdb'),bb,sequence)
                np.savez_compressed(output/f'{row["id"]}-{method}.npz',backbone=bb)
                path.write_text(json.dumps(record,indent=2))
                tracker.log({method+'/'+row['mode']+'/'+k:v for k,v in record.items() if isinstance(v,(int,float))},index)
                print('SCORED',json.dumps(record),flush=True)
        tracker.log({'compute/process_GPU_hours':(time.monotonic()-started)/3600},len(jobs))
        import wandb
        artifact=wandb.Artifact('noise-matched-generation-records',type='evaluation')
        for path in [*output.glob('*-native.json'),*output.glob('*-omni.json')]:
            artifact.add_file(str(path))
        artifact.add_file(str(ROOT/'benchmark_manifest.json'))
        artifact.add_file(str(provenance_path))
        tracker.run.log_artifact(artifact)
        (output/'generation_complete.json').write_text(json.dumps(dict(folding_predictions=80,scaffolds=48,checkpoint_sha256=ckpt_hash)))
        ok=True
    finally:tracker.finish(ok)


if __name__=='__main__':main()
