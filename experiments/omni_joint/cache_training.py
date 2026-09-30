"""Observed-only static contexts for joint training; no task flag enters the model."""
import argparse
import hashlib
import json
import random
import os
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals
from boltzgen.task.predict.data_from_yaml import Dataset, PredictionDataset, collate
from poc_model import ROOT, ASSETS, load_model, move, labels
from prompt import compile_prompt
from model import capture_context

EXP = ROOT/'experiments/omni_joint'
MODES = ('folding','design','inverse','mixed')


def ranges(mask):
    indices = np.where(mask)[0]+1
    return ','.join(map(str,indices))


def masks(row,mode):
    n = row['length']
    rng = random.Random(hashlib.sha256((row['id']+':'+mode).encode()).hexdigest())
    known_a,known_x = np.zeros(n,bool),np.zeros(n,bool)
    if mode=='folding': known_a[:] = True
    elif mode=='inverse': known_x[:] = True
    elif mode=='design':
        start = rng.randrange(3,n-20)
        known_a[start:start+16] = True
        known_x[start:start+16] = True
    elif mode=='mixed':
        a_start,x_start = rng.randrange(n-20),rng.randrange(n-20)
        known_a[a_start:a_start+20] = True
        known_x[x_start:x_start+20] = True
    return known_a,known_x


def build_entry(base,row,mode,split,canonicals=None,observed_backbone_only=False):
    output=EXP/'cache'; output.mkdir(exist_ok=True)
    yaml_dir=EXP/'prompts'; yaml_dir.mkdir(exist_ok=True)
    destination=output/f'{row["id"]}-{mode}.pt'
    if canonicals is None: canonicals=load_canonicals(str(ASSETS/'mols.zip'))
    known_a,known_x=masks(row,mode)
    spec={'path':str(Path(row['directory'])/'native.pdb'),
          'include':[{'chain':{'id':'A'}}],
          'structure_groups':[{'group':{'id':'A','visibility':0}}]}
    if known_x.any(): spec['structure_groups'].append({'group':{'id':'A','visibility':1,'res_index':ranges(known_x)}})
    if (~known_a).any(): spec['design']=[{'chain':{'id':'A','res_index':ranges(~known_a)}}]
    yaml_path=yaml_dir/f'{row["id"]}-{mode}.yaml'
    yaml_path.write_text(yaml.safe_dump({'entities':[{'file':spec}]},sort_keys=False))
    seed=int.from_bytes(hashlib.sha256((row['id']+mode).encode()).digest()[:4],'little')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset=PredictionDataset(Dataset(str(yaml_path),Tokenizer(atomize_modified_residues=False),Featurizer()),
        canonicals,str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False)
    default_rng=np.random.default_rng
    np.random.default_rng=lambda value=None:default_rng(seed if value is None else value)
    try: features=move(collate([dataset[0]]),'cuda')
    finally: np.random.default_rng=default_rng
    lab=labels(features)
    a=torch.tensor(known_a,device='cuda')[None]
    assert torch.equal(~features['design_mask'].bool(),a)
    atom_visible=torch.bmm(features['atom_to_token'].float(),torch.tensor(known_x,device='cuda').float()[None,:,None]).squeeze(-1).bool()
    atom_known=torch.bmm(features['atom_to_token'].float(),a.float()[...,None]).squeeze(-1).bool()
    hard=atom_visible & (atom_known | features['backbone_mask'].bool()) & features['atom_resolved_mask'].bool()
    if observed_backbone_only:
        hard = hard & features['backbone_mask'].bool()
    target=features['coords'].reshape(1,-1,3).clone()
    prompt=compile_prompt(base,features,observed_atoms=hard,canonical_frame=True)
    # Change hidden coordinates, including the full target-only tensors.
    other={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in features.items()}
    ox=other['coords'].reshape_as(target)
    ox[~hard]=12345
    counter=compile_prompt(base,other,observed_atoms=hard,canonical_frame=True)
    differences=[k for k,v in prompt.items() if isinstance(v,torch.Tensor) and not torch.equal(v,counter[k])]
    assert not differences,('hidden-coordinate leakage',differences)
    # Move the supervision into the same frame as declared hard observations.
    observed_x=prompt['coords'].reshape_as(target)
    if hard.any():
        first_atom=int(hard[0].nonzero()[0])
        target=target+(observed_x[:,first_atom:first_atom+1]-target[:,first_atom:first_atom+1])
        target=torch.where(hard[...,None],observed_x,target)
    context=capture_context(base,prompt)
    context['native_replay_eligible']=mode=='folding'
    # Store only legitimate observations in the sampler object; targets separately.
    identity=lab.clone(); identity[~a]=20
    payload=dict(cache=context,target=target,labels=lab,resolved=features['atom_resolved_mask'].float(),
                 observations=dict(identity=identity,identity_mask=a,coordinates=target.masked_fill(~hard[...,None],0),coordinate_mask=hard),
                 split=split,mode=mode,id=row['id'],hidden_coordinate_audit=True)
    torch.save(move(payload,'cpu'),str(destination)+f'.{os.getpid()}.tmp')
    Path(str(destination)+f'.{os.getpid()}.tmp').replace(destination)
    return destination


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--refresh',action='store_true')
    ap.add_argument('--index-only',action='store_true')
    args=ap.parse_args()
    started=time.monotonic()
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    records=json.loads((EXP/'manifests/data_filtered.json').read_text())
    base=None if args.index_only else load_model()
    canonicals=None if args.index_only else load_canonicals(str(ASSETS/'mols.zip'))
    output=EXP/'cache'; output.mkdir(exist_ok=True)
    yaml_dir=EXP/'prompts'; yaml_dir.mkdir(exist_ok=True)
    index=[]
    measurements=[]
    for split in ('train','validation'):
        for row in records[split][:args.limit or None]:
            for mode in MODES:
                destination=output/f'{row["id"]}-{mode}.pt'
                entry=dict(path=str(destination),id=row['id'],split=split,mode=mode)
                index.append(entry)
                if args.index_only: continue
                if destination.exists() and not args.refresh: continue
                torch.cuda.synchronize(); prompt_started=time.monotonic()
                build_entry(base,row,mode,split,canonicals)
                torch.cuda.synchronize()
                measurements.append(dict(seconds=time.monotonic()-prompt_started,tokens=row['length'],
                                         bytes=destination.stat().st_size))
                print('CACHE',split,row['id'],mode,flush=True)
    (EXP/'manifests/cache.json').write_text(json.dumps(index,indent=2))
    print('CACHE_COMPLETE',len(index),flush=True)
    if args.limit:
        (EXP/'manifests/cache_profile.json').write_text(json.dumps(dict(
            prompts=len(index),seconds=time.monotonic()-started,
            measurements=measurements,
            bytes=sum(Path(r['path']).stat().st_size for r in index)),indent=2))


if __name__=='__main__':main()
