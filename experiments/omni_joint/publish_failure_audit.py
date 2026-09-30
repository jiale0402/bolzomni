"""Publish recovered numerical controls without repeating model computation."""
import json
from pathlib import Path
import wandb
from tracking import Tracker,file_hash
EXP=Path(__file__).resolve().parent
sampler=json.loads((EXP/'results/sampling_controls.json').read_text())
components=json.loads((EXP/'results/component_controls.json').read_text())
assert len(sampler)==8 and len(components)==6
assert {r['variant'] for r in sampler}=={'native','zero_hard','trained_original','trained_late'}
assert {r['variant'] for r in components}=={'no_lora','no_feedback','all_masked'}
tracker=Tracker(EXP/'runs/consolidated-failure-audit',dict(validation_only=True,
                numerical_jobs=[7644594,7644983,7645831],
                component_job_publication_failed_and_recovered=True),
                'consolidated-failure-audit',job_type='diagnostic')
artifact=wandb.Artifact('omni-failure-audit',type='diagnostic')
for name in ('failure_diagnosis.json','sampling_controls.json','component_controls.json'):
    artifact.add_file(str(EXP/'results'/name))
artifact.add_file(str(EXP/'results/FAILURE_ANALYSIS.md'))
artifact.add_file(str(EXP/'results/diagnose_components_executed.py'))
tracker.run.log_artifact(artifact)
for i,row in enumerate(sampler+components):
    tracker.log({row['id']+'/'+row['variant']+'/'+k:v for k,v in row.items() if isinstance(v,(int,float))},i)
tracker.finish()
print('AUDIT_PUBLICATION_COMPLETE',flush=True)
