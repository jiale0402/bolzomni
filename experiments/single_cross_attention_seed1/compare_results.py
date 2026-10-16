"""Two-model comparison with real denominators; validation alone selects weights."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
METRICS=[('lnr','structural/AAR',100,True,2),('lnr','structural/complex_RMSD',1,False,3),
 ('rabd','structural/AAR',100,True,2),('rabd','structural/complex_RMSD',1,False,3),
 ('crossdocked2020','vina/dock',1,False,3),('crossdocked2020','chemical/qed',1,True,3),
 ('crossdocked2020','success',100,True,2)]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def value(event,key):
    if key=='success':return float(event['success'])
    item=event.get('scores')
    for k in key.split('/'):item=item.get(k) if isinstance(item,dict) else None
    return float(item) if isinstance(item,(float,int)) and math.isfinite(item) else float('nan')


def events(root,cfg):
    complete=read(root/'COMPLETE.json')
    assert complete['run_hash']==cfg['run_hash']
    assert sha(root/'REPORT.json')==complete['report_sha256']
    mapping={};counts={}
    for spec in cfg['benchmark_specs']:
        rows=[json.loads(line) for line in Path(spec['manifest']).read_text().splitlines()]
        assert sha(Path(spec['manifest']))==spec['manifest_sha256'] and len(rows)==spec['targets']
        for row in rows:
            path=root/'scores'/spec['name']/row['id']
            receipt=read(path/'SCORING_COMPLETE.json')
            assert receipt['checkpoint_sha256']==cfg['checkpoint_sha256']
            assert receipt['run_hash']==cfg['run_hash']
            values=[json.loads(line) for line in (path/'evaluation.jsonl').read_text().splitlines()]
            assert len(values)==30 and sorted(x['n'] for x in values)==list(range(30))
            for name,digest in receipt['score_files'].items():assert sha(path/name)==digest
            for event in values:
                key=(spec['name'],row['id'],event['n']);assert key not in mapping
                mapping[key]=event
        counts[spec['name']]=len(rows)
    assert len(mapping)==7590 and sum(counts.values())==253
    return mapping


def main():
    cfg=read(ROOT/'benchmark/run.json');ref=read(ROOT/'reference_benchmark_config.json')
    train=read(ROOT/'training/COMPLETE.json');reftrain=read(ROOT/'REFERENCE_TRAINING.json')
    assert sha(Path(cfg['checkpoint']))==train['checkpoint_sha256']==cfg['checkpoint_sha256']
    assert sha(Path(ref['checkpoint']))==reftrain['checkpoint_sha256']==ref['checkpoint_sha256']
    assert cfg['benchmark_specs']==ref['benchmark_specs']
    for key in ['seed','samples','batch_size','gamma']:assert cfg[key]==ref[key]
    old=events(Path(ref['output']),ref);new=events(ROOT/'benchmark',cfg)
    assert set(old)==set(new)
    rng=np.random.default_rng(20261010)
    rows=[]
    for bench,key,scale,higher,decimals in METRICS:
        targets=sorted({k[1] for k in new if k[0]==bench})
        arrays=[np.array([[value(m[(bench,t,n)],key) for n in range(30)] for t in targets]) for m in [old,new]]
        means=[float(np.nanmean(a))*scale if np.isfinite(a).any() else None for a in arrays]
        counts=[int(np.isfinite(a).sum()) for a in arrays]
        # Paired target bootstrap retains missing-score coverage of each model.
        bootstrap=[]
        if all(counts):
            for indices in rng.integers(0,len(targets),size=(20000,len(targets))):
                av,bv=(a[indices] for a in arrays)
                if np.isfinite(av).any() and np.isfinite(bv).any():bootstrap.append((np.nanmean(bv)-np.nanmean(av))*scale)
        ci=np.quantile(bootstrap,[.025,.975]).tolist() if bootstrap else None
        rows.append({'benchmark':bench,'metric':key,'higher_is_better':higher,'display_decimals':decimals,
            'targets':len(targets),'attempts':len(targets)*30,'method':means[0],'single_cross_attention':means[1],
            'method_finite_scores':counts[0],'baseline_finite_scores':counts[1],
            'method_missing':len(targets)*30-counts[0],'baseline_missing':len(targets)*30-counts[1],
            'paired_target_bootstrap_difference_ci95':ci})
    report={'completed':True,'models':{'method':ref['checkpoint_sha256'],'single_cross_attention':cfg['checkpoint_sha256']},
        'training_seeds':{'method':20261007,'baseline':1},'paired_evaluation_seed':cfg['seed'],
        'training_gpu':train['gpu'],'training_budget_seconds':14400,
        'method_actual_seconds':reftrain['job_elapsed_seconds_before_bookkeeping'],
        'baseline_actual_seconds':train['job_elapsed_seconds_before_bookkeeping'],
        'metrics':rows,'validation':{'method':reftrain['validation'],'baseline':train['validation']},
        'known_training_pdb_overlap':19,'benchmark_previously_inspected':True,
        'parameter_counts':{'method':4027712,'baseline':4028016},'verified_targets':253,'verified_attempts':7590,
        'ranking_does_not_imply_significance':True}
    # Report the selected checkpoints' actual holdout results; never select on them.
    for label,base,record in [('baseline',ROOT,train),('method',Path(ref['checkpoint']).parents[1],reftrain)]:
        path=base/'holdout'/f"step_{record['optimizer_step']:06d}"/'COMPLETE.json'
        holdout=read(path)
        # Evaluation snapshots omit optimizer state and have a different file
        # hash from best_joint.pt. Verify their exact fusion tensor contents.
        import torch
        snapshot=Path(holdout['protocol']['checkpoint'])
        assert sha(snapshot)==holdout['protocol']['checkpoint_sha256']
        selected=Path(cfg['checkpoint'] if label=='baseline' else ref['checkpoint'])
        a=torch.load(snapshot,map_location='cpu',weights_only=False)['fusion']
        b=torch.load(selected,map_location='cpu',weights_only=False)['fusion']
        assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
        report.setdefault('benchmark_holdout_loss',{})[label]=holdout['results']
    out=ROOT/'comparison';out.mkdir(exist_ok=True)
    (out/'RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    text=['# Single cross-attention vs multi-layer method','',
        'Training: one A100, 4 hours, same 30k manifest. Training seeds: baseline 1, historical method 20261007.',
        'Evaluation: unchanged paired seeds; 253 targets x 30 attempted slots. Best values are bold, second values underlined; ties share rank.',
        'Rank formatting does not establish statistical significance. Benchmark was previously inspected and has 19 known ligand training-PDB overlaps.','',
        '| Benchmark / metric | Method | Single cross-attention | Finite scores (method / baseline) |',
        '|---|---:|---:|---:|']
    for r in rows:
        vals=[r['method'],r['single_cross_attention']];d=r['display_decimals']
        shown=[round(x,d) for x in vals if x is not None]
        ranks=sorted(set(shown),reverse=r['higher_is_better'])
        cells=[]
        for x in vals:
            if x is None:cells.append('missing');continue
            label=f'{x:.{d}f}';rank=ranks.index(round(x,d))
            cells.append('**'+label+'**' if rank==0 else '<u>'+label+'</u>')
        text.append(f"| {r['benchmark']} / {r['metric']} | {cells[0]} | {cells[1]} | {r['method_finite_scores']} / {r['baseline_finite_scores']} |")
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    sys.path.insert(0,str(ROOT/'code'))
    from common import Tracker
    tracker=Tracker('comparison',{'training_seed':1,'protocol_unchanged':True})
    tracker.log({'comparison/targets':253,'comparison/attempts':7590,'comparison/completed':True})
    tracker.finish(report,[out/'RESULTS.json',out/'RESULTS.md'])
    (out/'COMPLETE.json').write_text(json.dumps({'passed':True,'report_sha256':sha(out/'RESULTS.json')},indent=2)+'\n')
    print(out/'RESULTS.md')


if __name__=='__main__':main()
