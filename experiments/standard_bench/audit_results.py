"""Verify published denominators against every official sequence score, and Slurm use."""
import csv
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
manifest=json.loads((ROOT/'manifest.json').read_text())
design=json.loads((OUT/'design_scores.json').read_text())
sequence_count=0
for row in design:
    for scaffold in row['scaffolds']:
        path=OUT/'motifbench'/row['method']/row['case']/scaffold['id']/'self_consistency/esm_eval_results.csv'
        seqs=list(csv.DictReader(path.open()))
        assert len(seqs)==8,path
        success=any(float(r['rmsd'])<2 and float(r['motif_rmsd'])<1 for r in seqs)
        assert success==scaffold['success']
        sequence_count+=len(seqs)
    assert len(row['scaffolds'])==2
    assert row['success']==sum(r['success'] for r in row['scaffolds'])
assert len(design)==12 and sequence_count==192
for row in manifest['folding']+manifest['design']:
    for method in ('native','omni'):
        record=json.loads((OUT/f'{row["id"]}-{method}.json').read_text())
        assert record['finite_geometry']
(OUT/'verification.json').write_text(json.dumps(dict(passed=True,folding_predictions=20,
    design_scaffolds=24,official_refolded_sequences=192,no_dropped_scaffolds=True),indent=2))
ids=['7658731','7659394','7660604']
raw=subprocess.check_output(['sacct','-j',','.join(ids),'-n','-P',
    '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
records=[]
for line in raw.splitlines():
    fields=line.split('|')
    if not fields or fields[0] not in ids:continue
    job,state,seconds,tres=fields[:4]
    gpus=int(dict(s.split('=',1) for s in tres.split(','))['gres/gpu'])
    records.append(dict(job_id=job,state=state,elapsed_seconds=int(seconds),gpus=gpus,allocation=tres))
assert {r['job_id'] for r in records}==set(ids)
assert all(r['state']=='COMPLETED' for r in records if r['job_id']!='7658731')
compute=dict(total_GPU_hours=sum(r['elapsed_seconds']*r['gpus'] for r in records)/3600,
    source='Slurm allocated GPU wall time; includes interrupted first generation job; excludes queue time and CPU-only dependency setup.',
    jobs=records)
(OUT/'compute_usage.json').write_text(json.dumps(compute,indent=2))
print('ALL_192_OFFICIAL_SCORES_VERIFIED',json.dumps(compute),flush=True)
