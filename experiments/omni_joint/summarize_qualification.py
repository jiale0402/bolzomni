"""Summarize validation diagnostics and publish their actual values to W&B."""
import json
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from tracking import Tracker

EXP=Path(__file__).resolve().parent


def aligned_rmsd(moving,reference):
    moving,reference=np.asarray(moving,dtype=float),np.asarray(reference,dtype=float)
    x,y=moving-moving.mean(0),reference-reference.mean(0)
    u,_,vt=np.linalg.svd(x.T@y)
    d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
    return float(np.sqrt(((x@(u@d@vt)-y)**2).sum(-1).mean()))


def main():
    root=EXP/'results/baseline-validation'
    records=json.loads((root/'native_metrics.json').read_text())
    parser=PDBParser(QUIET=True)
    for record in records:
        if record['mode']!='design':continue
        name=record['id']+'-design'
        pdbs=list((root/'af2').glob(name+'_unrelaxed_rank_001_*.pdb'))
        scores=list((root/'af2').glob(name+'_scores_rank_001_*.json'))
        if len(pdbs)!=1 or len(scores)!=1: raise RuntimeError('Missing/ambiguous AF2 result: '+name)
        residues=[r for r in parser.get_structure(name,pdbs[0])[0].get_residues() if r.id[0]==' ']
        bb=np.array([[r[a].coord for a in ('N','CA','C','O')] for r in residues])
        design=np.load(root/(name+'.npz'))
        assert bb.shape==design['backbone'].shape
        motif=design['motif']
        sc=aligned_rmsd(bb.reshape(-1,3),design['backbone'].reshape(-1,3))
        mr=aligned_rmsd(bb[motif].reshape(-1,3),design['native_backbone'][motif].reshape(-1,3))
        pae=float(np.array(json.loads(scores[0].read_text())['pae']).mean())
        record.update(af2_backbone_rmsd=sc,af2_motif_rmsd=mr,af2_mean_pae=pae,
                      design_pass=bool(record['sequence_valid'] and sc<2 and mr<1 and pae<5))
    destination=root/'qualification_summary.json'
    destination.write_text(json.dumps(records,indent=2))
    tracker=Tracker(EXP/'runs/native-validation',
                    {'purpose':'validation-baseline-qualification','formal_test':False,
                     'af2_model':'model_4_ptm','af2_recycles':3,'msa':'single_sequence',
                     'rmsd_atoms':['N','CA','C','O'],'rmsd_alignment':'independent least-squares fits'},
                    'native-baseline-validation',job_type='validation')
    for i,record in enumerate(records):
        values={f'baseline/{record["mode"]}/{k}':v for k,v in record.items() if isinstance(v,(int,float))}
        tracker.log(values,i)
    tracker.run.summary['folding_mean_CA_lDDT']=float(np.mean([r['ca_lddt'] for r in records if r['mode']=='folding']))
    tracker.run.summary['design_successes']=sum(r.get('design_pass',False) for r in records)
    tracker.run.summary['n_design_validation_samples']=sum(r['mode']=='design' for r in records)
    import wandb
    artifact=wandb.Artifact('native-validation-diagnostics',type='evaluation')
    artifact.add_file(str(destination)); tracker.run.log_artifact(artifact)
    tracker.finish()
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
