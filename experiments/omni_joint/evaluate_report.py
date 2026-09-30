"""Fixed paired statistics; failures stay in denominators, targets are clustered."""
import json
import subprocess
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from metrics import paired_summary
from summarize_qualification import aligned_rmsd
from tracking import Tracker,file_hash

EXP=Path(__file__).resolve().parent

def folding_clusters(rows):
    work=EXP/'results/test-clusters';work.mkdir(exist_ok=True)
    fasta=work/'test.fasta'
    fasta.write_text(''.join('>'+r['id']+'\n'+r['sequence']+'\n' for r in rows))
    hits=work/'hits.tsv'
    command=[str(EXP/'overlap/mmseqs/bin/mmseqs'),'easy-search',str(fasta),str(fasta),str(hits),str(work/'tmp'),
             '--min-seq-id','0.3','-c','0.8','--cov-mode','2','-s','7.5','--alignment-mode','3',
             '--format-output','query,target','--threads','4']
    with open(work/'search.log','w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    parent={r['id']:r['id'] for r in rows}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for line in hits.read_text().splitlines():
        a,b=line.split('\t');parent[root(a)]=root(b)
    mapping={key:root(key) for key in parent}
    (work/'clusters.json').write_text(json.dumps(mapping,indent=2))
    return mapping

def main():
    output=EXP/'results/paired-test'
    rows=json.loads((EXP/'manifests/data_filtered.json').read_text())['test']
    motifs=json.loads((EXP/'manifests/motifs.json').read_text())
    tasks=json.loads((EXP/'motif_tasks.json').read_text())['tasks']
    clusters=folding_clusters(rows)
    parser=PDBParser(QUIET=True);folding={};design={};raw=[]
    for method in ('native','omni'):
        folding[method]=[json.loads((output/(r['id']+'-'+method+'.json')).read_text()) for r in rows]
        design[method]={t['id']:[] for t in tasks}
        for row in motifs:
            stem=row['task']+'-c'+str(row['candidate']).zfill(2)+'-'+method
            record=json.loads((output/(stem+'.json')).read_text())
            record['design_pass']=False
            qualified=record['sequence_valid'] and record['finite_geometry'] and record['identity_geometry_consistent']
            if qualified:
                pdbs=list((output/'af2').glob(stem+'_unrelaxed_rank_001_*.pdb'))
                scores=list((output/'af2').glob(stem+'_scores_rank_001_*.json'))
                assert len(pdbs)==len(scores)==1,('Missing independent refold',stem)
                residues=[r for r in parser.get_structure(stem,pdbs[0])[0].get_residues() if r.id[0]==' ']
                backbone=np.asarray([[r[a].coord for a in ('N','CA','C','O')] for r in residues])
                generated=np.load(output/(stem+'.npz'));mask=generated['motif']
                assert backbone.shape==generated['backbone'].shape
                sc=aligned_rmsd(backbone.reshape(-1,3),generated['backbone'].reshape(-1,3))
                mr=aligned_rmsd(backbone[mask].reshape(-1,3),generated['native_backbone'][mask].reshape(-1,3))
                pae=float(np.asarray(json.loads(scores[0].read_text())['pae']).mean())
                record.update(af2_backbone_rmsd=sc,af2_motif_rmsd=mr,af2_mean_pae=pae,
                              design_pass=bool(sc<2 and mr<1 and pae<5))
            design[method][row['task']].append(record);raw.append(record)
    fold_report={metric:paired_summary([r[metric] for r in folding['native']],[r[metric] for r in folding['omni']],
                    groups=[clusters[r['id']] for r in rows]) for metric in ('ca_lddt','tm_score')}
    # RMSD is a lower-is-better descriptive secondary, not a positivity criterion.
    if all(r['ca_rmsd'] is not None for group in folding.values() for r in group):
        fold_report['ca_rmsd']=paired_summary([r['ca_rmsd'] for r in folding['native']],
                          [r['ca_rmsd'] for r in folding['omni']],groups=[clusters[r['id']] for r in rows])
        fold_report['ca_rmsd']['positive']=bool(fold_report['ca_rmsd']['ci95'][1]<0)
    else:
        fold_report['ca_rmsd']={'unavailable':'Nonfinite predictions; primary lDDT failures scored zero, no dropped targets.'}
    rates={method:[sum(r['design_pass'] for r in design[method][task['id']])/16 for task in tasks]
           for method in ('native','omni')}
    design_report=paired_summary(rates['native'],rates['omni'],groups=[t['pdb'] for t in tasks])
    design_report['per_task']={t['id']:{m:rates[m][i] for m in rates} for i,t in enumerate(tasks)}
    for method in ('native','omni'):
        values=[r for group in design[method].values() for r in group]
        assert len(values)==400
        design_report[method+'_counts']=dict(total=len(values),successes=sum(r['design_pass'] for r in values),
                    invalid_sequences=sum(not r['sequence_valid'] for r in values),
                    identity_geometry_failures=sum(not r['identity_geometry_consistent'] for r in values),
                    solved_tasks=sum(any(r['design_pass'] for r in group) for group in design[method].values()),
                    unique_successful_sequences=len({r['sequence'] for r in values if r['design_pass']}))
    summary=dict(folding=fold_report,design=design_report,
                 joint_positive=fold_report['ca_lddt']['positive'] and design_report['positive'],
                 conclusion_scope='One training seed; one shared checkpoint; protein folding and own-sequence motif design only.',
                 feedback_causal_attribution_tested=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'design_evaluations.json').write_text(json.dumps(raw,indent=2))
    tracker=Tracker(output/'report-tracking',dict(checkpoint_sha256=file_hash(EXP/'runs/joint-seed-1729/selected.pt'),
                        report_source_sha256=file_hash(__file__),formal_test=True,bootstrap_replicates=10000),
                    'paired-final-comparison',job_type='evaluation')
    tracker.log(dict(folding_lddt_difference=fold_report['ca_lddt']['difference'],
                     folding_lddt_ci_lower=fold_report['ca_lddt']['ci95'][0],
                     design_success_difference=design_report['difference'],
                     design_success_ci_lower=design_report['ci95'][0],joint_positive=summary['joint_positive']),0)
    import wandb
    artifact=wandb.Artifact('paired-omni-comparison',type='evaluation')
    for name in ('summary.json','design_evaluations.json'):artifact.add_file(str(output/name))
    tracker.run.log_artifact(artifact);tracker.finish()
    print('FINAL_COMPARISON',json.dumps(summary),flush=True)

if __name__=='__main__':main()
