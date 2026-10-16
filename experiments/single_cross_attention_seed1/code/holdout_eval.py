"""Paired real-reference benchmark losses for reporting, never selection."""
import hashlib
import json
from pathlib import Path
import time
import torch
from common import ROOT, sha, atomic_json
from learning import BaseDataset
from native_checks import fixed_rng

HOLDOUT_ROOT=Path('/projects/p53127/anewomni_work/trimodal_holdout_loss_20261010')
MANIFEST_SHA='597ae47a91c17c47638541c5d97423e639221d95b325a8082e0ffcda1dd066ba'
METRICS=('H','X','total')

def evaluate(model,epsilon,step,checkpoint,tracker):
    start=time.monotonic()
    assert sha(HOLDOUT_ROOT/'manifest.json')==MANIFEST_SHA
    rows=json.loads((HOLDOUT_ROOT/'manifest.json').read_text())
    assert len(rows)==253
    out=ROOT/'holdout'/f'step_{step:06d}'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'STARTED.json').exists(),'Inspect interrupted holdout evaluation before repeating it'
    protocol={'optimizer_step':step,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
              'manifest_sha256':MANIFEST_SHA,'targets':253,'draws':4,'selection_uses_holdout':False,
              'known_ligand_training_pdb_overlap':19,'training_updates':0}
    atomic_json(out/'STARTED.json',protocol)
    sums={m:dict.fromkeys(METRICS,0.) for m in ['peptide','antibody','ligand']}
    blocks=dict.fromkeys(sums,0);counts=dict.fromkeys(sums,0)
    was_training=model.training
    versions={n:p._version for n,p in model.named_parameters()}
    model.eval()
    try:
        with torch.inference_mode(),(out/'records.jsonl').open('w') as handle:
            for row in rows:
                assert sha(row['input_path'])==row['input_sha256']
                assert sha(row['feature_path'])==row['feature_sha256']
                saved=torch.load(row['input_path'],map_location='cpu',weights_only=False)
                f=torch.load(row['feature_path'],map_location='cpu',weights_only=True)
                f={'single':f['single'],'pair':f['pair'],'mask':saved['coverage'],
                   'generate_mask':saved['data']['generate_mask']}
                epsilon.set_features([{k:v.cuda() for k,v in f.items()}])
                target_values=[]
                for draw in range(4):
                    value=int.from_bytes(hashlib.sha256(f"holdout:20261010:{row['key']}:{draw}".encode()).digest()[:4],'big')
                    data={k:v.clone() if torch.is_tensor(v) else v for k,v in saved['data'].items()}
                    batch={k:v.cuda() for k,v in BaseDataset.collate_fn(None,[data]).items()}
                    with fixed_rng(value): result=model(**batch)
                    assert isinstance(result,dict) and all(torch.isfinite(result[k]) for k in METRICS)
                    result={k:float(result[k]) for k in METRICS}
                    assert abs(result['total']-(model.h_loss_weight*result['H']+result['X']))<2e-5
                    target_values.append(result)
                    handle.write(json.dumps({'target':row['target'],'key':row['key'],'modality':row['modality'],
                                             'draw':draw,'seed':value,'generated_blocks':row['generated_blocks'],**result})+'\n')
                m=row['modality'];n=row['generated_blocks']
                for k in METRICS:sums[m][k]+=sum(v[k] for v in target_values)*n/4
                blocks[m]+=n;counts[m]+=1
        assert counts=={'peptide':93,'antibody':60,'ligand':100}
        assert versions=={n:p._version for n,p in model.named_parameters()}
        results={m:{k:sums[m][k]/blocks[m] for k in METRICS}|
                 {'records':counts[m],'generated_blocks':blocks[m],'noise_evaluations':counts[m]*4} for m in sums}
        results['combined']={k:sum(s[k] for s in sums.values())/sum(blocks.values()) for k in METRICS}
        report={'completed':True,'protocol':protocol,'results':results,'seconds':time.monotonic()-start,
                'records_sha256':sha(out/'records.jsonl'),'gpu':torch.cuda.get_device_name(),
                'parameters_unchanged':True}
        atomic_json(out/'COMPLETE.json',report)
        tracker.log({'optimizer_step':step,**{f'holdout/{m}/{k.lower()}':v for m,vs in results.items() for k,v in vs.items()},
                     'holdout/checkpoint_sha256':protocol['checkpoint_sha256'],'holdout/evaluation_seconds':report['seconds']})
        return report
    finally:
        model.train(was_training);model.autoencoder.eval()
