"""Native BoltzGen trunk only; cache compact covered-block single/pair features."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import gc
import json
import time
import numpy as np
import torch
from runtime import ROOT, ASSETS, CONFIG, CACHE_ROOT, HERE, atomic_json, save_tensor, seed, sha, Tracker
from donor_input import prepare
from boltzgen.model.models.boltz import Boltz
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals
from boltzgen.task.predict.data_from_yaml import Dataset, PredictionDataset, collate


TRAINING_ROOT=ROOT
ROOT=CACHE_ROOT

class TrunkDone(Exception):pass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    rows=json.loads((ROOT/'manifests'/('profile_rows.json' if args.profile else 'feature_rows.json')).read_text())
    if not args.profile:rows=[r for r in rows if int(r['key'],16)%args.shards==args.shard]
    checkpoint=ASSETS/'boltzgen1_diverse.ckpt'
    provenance={'checkpoint_sha256':sha(checkpoint),'recycles':3,'compact':True,
        'source_hashes':{'extract_benchmark.py':sha(Path(__file__)),'donor_input.py':sha(HERE/'donor_input.py')}}
    stage='evaluation-feature-profile' if args.profile else f'evaluation-features-{args.shard}'
    tracker=Tracker(stage,provenance)
    model=Boltz.load_from_checkpoint(str(checkpoint),map_location='cpu',strict=True,weights_only=False,
        use_ema=False,validators=None,masker_args={'mask':True,'mask_backbone':False},
        predict_args={'recycling_steps':3,'sampling_steps':1,'diffusion_samples':1})
    model.eval().requires_grad_(False).cuda();model.use_kernels=False
    captured={};calls=0
    def hook(module,inputs,output):
        nonlocal calls
        calls+=1
        if calls==4:
            captured['s'],captured['z']=output[0].detach(),output[1].detach()
            raise TrunkDone()
    handle=model.pairformer_module.register_forward_hook(hook)
    canonical=load_canonicals(str(ASSETS/'mols.zip'))
    done=0;started=time.monotonic()
    for row in rows:
        key=row['key'];dest=ROOT/'features/aligned'/(key+'.pt');status=ROOT/'features/status'/(key+'.json')
        if dest.exists() and status.exists():
            prior=json.loads(status.read_text())
            assert prior['provenance']==provenance and sha(dest)==prior['sha256']
            done+=1;continue
        tick=time.monotonic();task_seed=int(key[:8],16);seed(task_seed)
        path,meta=prepare(ROOT,key)
        generated=torch.tensor(meta['generate_mask'],dtype=torch.bool)
        assert len(generated)==row['nblocks']
        calls=0;captured.clear();gc.collect();torch.cuda.empty_cache()
        if path is None:
            indices=torch.full((len(generated),),-1,dtype=torch.long)
            single=torch.empty((0,384));pair=torch.empty((0,0,128),dtype=torch.bfloat16)
            nt=0
        else:
            ds=PredictionDataset(Dataset([str(path)],Tokenizer(atomize_modified_residues=False),Featurizer()),
                canonical,str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False,
                extra_features=['structure','tokenized'])
            old_rng=np.random.default_rng
            np.random.default_rng=lambda value=None:old_rng(task_seed if value is None else value)
            try:feats=ds[0]
            finally:np.random.default_rng=old_rng
            tokens=feats.pop('tokenized').tokens;structure=feats.pop('structure')
            names={int(c['asym_id']):str(c['name']) for c in structure.chains}
            order=[(names[int(t['asym_id'])],int(t['res_idx'])) for t in tokens]
            assert len(order)==len(set(order))
            lookup={v:i for i,v in enumerate(order)}
            indices=torch.tensor([lookup[tuple(v)] if v is not None else -1 for v in meta['mapping']])
            expected=torch.zeros(len(order),dtype=torch.bool)
            for idx,gen in zip(indices,generated):
                if idx>=0 and gen:expected[idx]=True
            assert torch.equal(expected,feats['design_mask'].bool()),(key,'design mask mismatch')
            if row['modality']=='ligand':assert bool((indices[generated]<0).all())
            # Canonical design placeholder coordinates are zero in source and parsed structure.
            for token,gen in zip(tokens,expected):
                if gen:
                    atoms=structure.atoms[int(token['atom_idx']):int(token['atom_idx'])+int(token['atom_num'])]
                    assert np.allclose(atoms['coords'],0),(key,'generated coordinates exported')
            batch=collate([feats]);batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            with torch.inference_mode():
                masked=model.masker(batch)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    try:model(masked,recycling_steps=3,num_sampling_steps=1)
                    except TrunkDone:pass
            assert calls==4
            covered=indices[indices>=0].to(captured['s'].device)
            single=captured['s'][0,covered].float().cpu().contiguous()
            pair=captured['z'][0,covered[:,None],covered[None,:]].to(torch.bfloat16).cpu().contiguous()
            nt=len(order)
            del batch,masked,feats,ds
        assert torch.isfinite(single).all() and torch.isfinite(pair).all()
        record={'key':key,'single':single,'pair':pair,'mask':indices>=0,'generate_mask':generated,
            'donor_index':indices,'compact':True,'metadata':meta,'provenance':provenance}
        save_tensor(dest,record)
        coverage={'blocks':len(indices),'covered':int((indices>=0).sum()),
            'generated':int(generated.sum()),'covered_generated':int(((indices>=0)&generated).sum())}
        report={'key':key,'modality':row['modality'],'provenance':provenance,'sha256':sha(dest),
            'input_sha256':row['input_sha256'],'tokens':nt,'coverage':coverage,
            'seconds':time.monotonic()-tick,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}
        atomic_json(status,report);done+=1
        tracker.log({'completed':done,'total':len(rows),'seconds_per_record':report['seconds'],**coverage})
        print('FEATURE',stage,done,len(rows),key,nt,round(report['seconds'],2),flush=True)
        captured.clear()
    handle.remove()
    report={'complete':done,'expected':len(rows),'seconds':time.monotonic()-started,'provenance':provenance}
    atomic_json(ROOT/'features'/(stage+'.json'),report);tracker.finish(report)


if __name__=='__main__':main()
