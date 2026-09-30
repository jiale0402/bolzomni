"""Invoke pinned MotifBench refolding and success scoring, with identical fixed MPNN seed."""
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'vendor/MotifBench'
LAB=VENDOR/'Scaffold-Lab'
sys.path.insert(0,str(LAB))
os.chdir(LAB)
import esm
from motif_refolding import MotifRefolder
from analysis import utils as au
from tracking import Tracker,file_hash

def main():
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    output=ROOT/'results'
    tracker=Tracker(output/'refolding-tracking',dict(
        manifest_sha256=file_hash(ROOT/'manifest.json'),source_sha256=file_hash(__file__),
        evaluator_revision=manifest['motifbench_revision'],
        evaluator_source_sha256=file_hash(LAB/'motif_refolding.py'),
        success_source_sha256=file_hash(LAB/'analysis/utils.py'),
        sequence_count=8,mpnn_seed=2,sc_rmsd_cutoff=2.,motif_rmsd_cutoff=1.,
        notes='Official scoring unchanged; optional PyMOL disabled; cache ESMFold weights; fix ProteinMPNN subprocess seed.',
        training_performed=False),'standard-benchmark-subset-refolding',job_type='evaluation')
    # Load the exact same evaluator once, then reuse it across all cases/methods.
    original_loader=esm.pretrained.esmfold_v1
    shared=[]
    def cached_loader():
        if not shared:shared.append(original_loader())
        return shared[0]
    esm.pretrained.esmfold_v1=cached_loader
    # Official ProteinMPNN defaults to random seed=0. Fix its public CLI option
    # identically for both arms; no changes to its model or scoring functions.
    original_popen=subprocess.Popen
    def seeded_popen(args,*a,**kw):
        if isinstance(args,list) and any(str(x).endswith('/protein_mpnn_run.py') for x in args):
            args=args+['--seed','2']
        return original_popen(args,*a,**kw)
    subprocess.Popen=seeded_popen
    ok=False;started=time.monotonic();records=[]
    try:
        for index,row in enumerate(manifest['motif_cases']):
            for method in ('native','omni'):
                case=row['case'];scaffolds=output/'scaffolds'/method/case
                dest=output/'motifbench'/method/case;dest.mkdir(parents=True,exist_ok=True)
                info=dest/'motif_info.csv'
                reference=ROOT/'data/motifbench/motif_pdbs'/f'{case}.pdb'
                subprocess.run([sys.executable,str(VENDOR/'scripts/write_motifInfo_from_scaffoldInfo.py'),
                    str(scaffolds/'scaffold_info.csv'),str(reference),str(info)],check=True)
                subprocess.run([sys.executable,str(VENDOR/'scripts/check_segment_validity.py'),
                    '--input',str(ROOT/'data/motifbench/test_cases.csv'),'--test',str(info)],check=True)
                conf=OmegaConf.load(LAB/'config/motif_scaffolding.yaml')
                conf.inference.backbone_pdb_dir=str(scaffolds)
                conf.inference.output_dir=str(dest)
                conf.inference.motif_csv_path=str(info)
                conf.inference.motif_pdb=str(reference)
                conf.inference.samples.max_backbones=2
                conf.inference.samples.seq_per_sample=8
                conf.inference.samples.mpnn_batch_size=8
                conf.inference.force_motif_AA_type=True
                conf.evaluation.visualize=False
                random.seed(2);np.random.seed(2);torch.manual_seed(2);torch.cuda.manual_seed_all(2)
                begin=time.monotonic()
                refolder=MotifRefolder(conf)
                refolder.run_sampling()
                merged,count=au.csv_merge(dest,'esm')
                assert count==2 and len(merged)==16,(method,case,count,len(merged))
                complete,summary,success,successful,_=au.analyze_success_rate(merged)
                complete.to_csv(dest/'official_sequence_scores.csv',index=False)
                summary.to_csv(dest/'official_summary.csv',index=False)
                record=dict(method=method,case=case,group=row['group'],success=int(success),total=2,
                    success_rate=float(success/2),seconds=time.monotonic()-begin,
                    scaffolds=[dict(id=Path(p).stem,success=p in successful,
                        best_sc_rmsd=float(g.rmsd.min()),best_motif_rmsd=float(g.motif_rmsd.min()))
                        for p,g in complete.groupby('backbone_path')])
                (dest/'scores.json').write_text(json.dumps(record,indent=2))
                records.append(record)
                tracker.log({method+'/design/success_rate':record['success_rate'],
                    method+'/design/successes':int(success),method+'/design/case_index':index},index)
                print('MOTIFBENCH_SCORED',json.dumps(record),flush=True)
        (output/'design_scores.json').write_text(json.dumps(records,indent=2))
        import wandb
        artifact=wandb.Artifact('standard-benchmark-motifbench-scores',type='evaluation')
        artifact.add_file(str(output/'design_scores.json'))
        artifact.add_dir(str(output/'motifbench'),name='official_motifbench_outputs')
        tracker.run.log_artifact(artifact)
        tracker.log({'compute/GPU_hours':(time.monotonic()-started)/3600,
            'compute/peak_memory_GB':torch.cuda.max_memory_allocated()/1e9},6)
        ok=True
    finally:
        subprocess.Popen=original_popen
        esm.pretrained.esmfold_v1=original_loader
        tracker.finish(ok)

if __name__=='__main__':main()
