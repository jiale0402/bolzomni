"""Independent metrics/denominators, plots, accounting and W&B verification."""
import csv
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import wandb
from tracking import Tracker, file_hash
from final_audit import independent_ca_lddt, recompute_refold_metrics
import report

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OUT=ROOT/'results'
RUN=ROOT/'runs/native-1729'
OLD=ROOT.parent/'official_poc'
WB='jialechen2025-northwestern-university/boltz-omni-joint'

def read(p):return json.loads(p.read_text())

def main():
    lock=read(ROOT/'input_lock.json');manifest=read(ROOT/'benchmark_manifest.json')
    assert file_hash(ROOT/'benchmark_manifest.json')==lock['manifest_sha256']
    assert file_hash(OLD/'data_manifest.json')==lock['data_manifest_sha256']
    for f in lock['baseline']:assert file_hash(OUT/f['path'])==f['sha256']
    checkpoint=file_hash(RUN/'final.pt')
    cfg=read(RUN/'config.json')
    for name,expected in cfg['sources'].items():assert file_hash(PROJECT/name)==expected,name
    train=[read_json for line in (RUN/'metrics.jsonl').read_text().splitlines()
        if 'train/loss' in (read_json:=json.loads(line))]
    assert sorted(r['optimizer_step'] for r in train)==list(range(1,401))
    assert all(math.isfinite(v) for r in train for k,v in r.items() if isinstance(v,(float,int)))
    assert (RUN/'initial.pt').exists() and read(RUN/'completion.json')['training_completed']
    ca_errors=[];scored=0;sequences=0
    for row in manifest['folding']:
        for i in range(5):
            ident=f'{row["id"]}_{i}'
            a,b=(read(OUT/f'{ident}-{m}.json') for m in ('native','omni'))
            assert a['seed']==b['seed'] and b['checkpoint_sha256']==checkpoint
            for m in ('native','omni'):
                saved=read(OUT/f'{ident}-{m}.json')
                bb=np.load(OUT/f'{ident}-{m}.npz')['backbone']
                value=independent_ca_lddt(bb,row)
                ca_errors.append(abs(value-saved['ca_lddt']));scored+=1
    assert max(ca_errors)<1e-7
    design=read(OUT/'design_scores.json');assert len(design)==12
    rmsd_errors=[]
    for row in manifest['design']:
        a,b=(read(OUT/f'{row["id"]}-{m}.json') for m in ('native','omni'))
        assert a['seed']==b['seed'] and b['checkpoint_sha256']==checkpoint
        assert file_hash(Path(row['yaml']))==row['input_sha256']
        case=next(c for c in manifest['design_cases'] if c['case']==row['case'])
        assert file_hash(Path(case['reference']))==case['reference_sha256']
        for method in ('native','omni'):
            folder=OUT/'refold'/method/row['case']
            with (folder/'sequence_scores.csv').open() as stream:allseq=list(csv.DictReader(stream))
            selected=[s for s in allseq if Path(s['backbone_path']).stem==row['id']]
            assert len(selected)==8 and len({s['sample_idx'] for s in selected})==8
            passed=[]
            for seq in selected:
                actual=recompute_refold_metrics(seq,row,case)
                for key in ('rmsd','motif_rmsd'):rmsd_errors.append(abs(actual[key]-float(seq[key])))
                p=actual['rmsd']<=2 and actual['motif_rmsd']<=1 and float(seq['plddt'])>=70 and float(seq['pae'])<=5
                assert p==(seq['paper_success']=='True')
                passed.append(p);sequences+=1
            record=next(c for c in design if c['method']==method and c['case']==row['case'])
            scaffold=next(s for s in record['scaffolds'] if s['id']==row['id'])
            assert scaffold['success']==any(passed) and scaffold['sequence_count']==8
    assert sequences==384 and max(rmsd_errors)<1e-7
    report.main()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    steps=np.array([r['optimizer_step'] for r in train])
    for ax,keys,title in zip(axes.flat,[['train/folding/loss','train/design/loss'],
        ['train/folding/coordinate','train/design/coordinate'],
        ['train/design/identity','train/design/replay'],['train/gradient_norm_before_clip']],
        ['Joint objective by task','Coordinate loss by task','Design CE and retention MSE','Gradient norm before clipping']):
        for k in keys:
            y=np.array([r[k] for r in train]);ax.plot(steps,y,alpha=.18)
            ax.plot(steps[19:],np.convolve(y,np.ones(20)/20,mode='valid'),label=k.replace('train/',''))
        ax.set(title=title,xlabel='Update');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    submissions=[read(p) for p in (ROOT/'submissions').glob('*.json')]
    jobids=[r['job_id'].split(';')[0] for r in submissions]
    raw=subprocess.check_output(['sacct','-nP','-j',','.join(jobids),
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Account,NodeList'],text=True)
    accounting=[];gpu_hours=0.
    for line in raw.splitlines():
        parts=line.split('|')
        if parts[0] not in jobids:continue
        gpu=0
        for item in parts[4].split(','):
            if item.startswith('gres/gpu='):gpu=int(item.split('=')[1])
        gpu_hours+=gpu*int(parts[3])/3600
        accounting.append(dict(job_id=parts[0],state=parts[1],exit_code=parts[2],seconds=int(parts[3]),
            GPU_count=gpu,allocation=parts[5],node=parts[6]))
    assert len(accounting)==len(jobids)
    # Every failed attempt is included, whether it completed model work or not.
    for r in submissions:
        eligible=[a for a,v in r['eligibility'].items() if v['eligible']]
        assert r['user_fairshare'][r['account']]==max(r['user_fairshare'][a] for a in eligible)
    api=wandb.Api(timeout=45);sync=[]
    for directory in (RUN,OUT/'generation-tracking',OUT/'refold-tracking',OUT/'report-tracking'):
        status=read(directory/'tracking_status.json')
        assert status['mode']=='online' and not status['sync_required']
        remote=api.run(WB+'/'+status['id'])
        assert remote.state=='finished',(directory,remote.state)
        sync.append(dict(id=remote.id,state=remote.state,url=remote.url))
    audit=dict(complete=True,training_updates=400,folding_predictions=scored,scaffolds=48,refold_sequences=sequences,
        ca_lddt_max_error=max(ca_errors),rmsd_max_error=max(rmsd_errors),baseline_files_verified=len(lock['baseline']),
        one_final_checkpoint_sha256=checkpoint,initial_checkpoint_sha256=file_hash(RUN/'initial.pt'),
        allocated_GPU_hours=gpu_hours,jobs=accounting,wandb_runs=sync,summary_sha256=file_hash(OUT/'summary.json'))
    (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2))
    tracker=Tracker(OUT/'audit-tracking',dict(checkpoint_sha256=checkpoint,source_sha256=file_hash(__file__)),
        'native-condition-final-audit',job_type='evaluation')
    ok=False
    try:
        tracker.log({'audit/complete':1,'compute/allocated_GPU_hours_including_failures':gpu_hours},400)
        tracker.run.log({'training_curves':wandb.Image(str(OUT/'training_curves.png'))})
        artifact=wandb.Artifact('native-condition-complete-evidence',type='evaluation')
        for p in [ROOT/'RESULTS.md',ROOT/'PLAN.md',ROOT/'input_lock.json',OUT/'completion_audit.json',
            OUT/'summary.json',OUT/'folding_scores.csv',OUT/'design_scores.csv',OUT/'training_curves.png']:
            artifact.add_file(str(p))
        for p in (OUT/'refold').glob('*/*/sequence_scores.csv'):
            artifact.add_file(str(p),name=str(p.relative_to(OUT)))
        tracker.run.log_artifact(artifact);ok=True
    finally:tracker.finish(ok)
    print('AUDIT_COMPLETE',json.dumps(audit),flush=True)

if __name__=='__main__':main()
