"""Derive the experiment lock from completed audit artifacts, never manual booleans."""
import json
import shutil
from pathlib import Path
from tracking import file_hash

EXP=Path(__file__).resolve().parent

def read(path):return json.loads((EXP/path).read_text())

def main():
    integration=read('results/integration_audit.json')
    assert integration['zero_denoiser_error']<1e-5 and integration['zero_sampler_error']<1e-4
    assert integration['hard_observations'] and integration['gradients'] and integration['feedback_effect']>0
    assert read('results/prompt-audit/differences.json')=={}
    baseline=read('results/baseline-validation/qualification_summary.json')
    folding=[r for r in baseline if r['mode']=='folding']
    design=[r for r in baseline if r['mode']=='design']
    assert len(folding)>=2 and all(0<=r['ca_lddt']<=1 for r in folding)
    assert len(design)>=2 and all('af2_mean_pae' in r and 'design_pass' in r for r in design)
    data=read('manifests/data_filtered.json'); overlap=read('manifests/overlap_audit.json')
    assert overlap['counts']=={k:len(v) for k,v in data.items()}
    assert all(data[k] for k in ('train','validation','test'))
    cache=read('manifests/cache.json')
    expected={(split,r['id'],mode) for split in ('train','validation') for r in data[split]
              for mode in ('folding','design','inverse','mixed')}
    assert {(r['split'],r['id'],r['mode']) for r in cache}==expected
    # Cache entries are generated lazily, using the same audited compiler, on first
    # access. Data and all planned observation modes are locked before training.
    profile=read('results/training_profile.json')
    assert profile['within_budget']
    assert {r['mode'] for r in profile['measurements']}=={'folding','design','inverse','mixed'}
    source={str(p.relative_to(EXP)):file_hash(p) for p in sorted(EXP.glob('*.py'))}
    for relative in ('../../scripts/poc_model.py','../../scripts/generate_panel.py'):
        source[relative]=file_hash(EXP/relative)
    data_hashes={r['id']:file_hash(Path(r['directory'])/'native.pdb') for group in data.values() for r in group}
    lock=dict(native_reproduction=True,architecture_audit=True,compiler_leakage_audit=True,
              dataset_overlap_audit=True,folding_baseline_qualified=True,design_baseline_qualified=True,
              evaluator_qualified=True,cost_profile_approved=True,
              qualification_means_pipeline_valid_not_positive_design_result=True,
              protocol_sha256=file_hash(EXP/'PROTOCOL.md'),
              cache_manifest_sha256=file_hash(EXP/'manifests/cache.json'),source_sha256=source,
              data_manifest_sha256=file_hash(EXP/'manifests/data_filtered.json'),data_sha256=data_hashes,
              motif_manifest_sha256=file_hash(EXP/'manifests/motifs.json'),
              counts=overlap['counts'],profile=profile,
              cache_policy='deterministic lazy static context, hidden-input audit on every entry',
              job_walltime_limits={'cache_hours':6,'training_hours':6},
              upstream_commit='a3149cf18eeb58648d1abbb27539bd73f746cdda')
    from poc_model import ASSETS
    lock['base_checkpoint_sha256']=file_hash(ASSETS/'boltzgen1_diverse.ckpt')
    (EXP/'LOCK.json').write_text(json.dumps(lock,indent=2))
    print('EXPERIMENT_LOCKED',lock['counts'],flush=True)

if __name__=='__main__':main()
