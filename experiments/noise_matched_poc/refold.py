"""ProteinMPNN(8) + ESMFold, using all four BoltzGen Appendix C.5 criteria."""
import argparse
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
VENDOR=ROOT.parent/'standard_bench/vendor/MotifBench'
LAB=VENDOR/'Scaffold-Lab'
sys.path.insert(0,str(LAB));os.chdir(LAB)
import esm
from motif_refolding import MotifRefolder
from analysis import utils as au
from tracking import Tracker,file_hash
from final_audit import recompute_refold_metrics
from raw_joint import design_pass


def refold_raw_sequences(manifest, output, model_loader, tracker):
    """At most 48 unrepaired sequences; invalid joint outputs count as failures.

This extra readout does not replace the locked best-of-8 MPNN benchmark.
It uses the identical pinned ESMFold and four numeric success thresholds.
"""
    raw=[];model=None
    for index,candidate in enumerate(manifest['design']):
        case=next(r for r in manifest['design_cases'] if r['case']==candidate['case'])
        for method in ('native','omni'):
            dest=output/'raw_refold'/method/candidate['case'];dest.mkdir(parents=True,exist_ok=True)
            path=dest/(candidate['id']+'.json')
            generated=json.loads((output/f'{candidate["id"]}-{method}.json').read_text())
            if path.exists():
                cached=json.loads(path.read_text())
                assert cached['checkpoint_sha256']==generated['checkpoint_sha256']
                assert cached['sequence']==generated['sequence']
                assert cached['validity']==generated['raw_validity']
                raw.append(cached);continue
            record=dict(id=candidate['id'],case=candidate['case'],method=method,
                sequence=generated['sequence'],checkpoint_sha256=generated['checkpoint_sha256'],
                validity=generated['raw_validity'],success=False,refolded=False,
                rmsd=None,motif_rmsd=None,pae=None,plddt=None)
            if not generated['raw_validity']['valid']:
                record['failure_reason']='Invalid or inconsistent unrepaired joint output'
            else:
                if model is None:model=model_loader().eval().cuda()
                random.seed(2);np.random.seed(2);torch.manual_seed(2);torch.cuda.manual_seed_all(2)
                folded=dest/(candidate['id']+'.pdb')
                with torch.no_grad():
                    result=model.infer(record['sequence'])
                    pdb=model.output_to_pdb(result)[0]
                    confidence=dict(pae=float(result['predicted_aligned_error'].mean()),
                        plddt=float(result['mean_plddt'].item()))
                folded.write_text(pdb)
                scaffold=output/'scaffolds'/method/candidate['case']/(candidate['id']+'.pdb')
                metrics=recompute_refold_metrics(dict(backbone_path=str(scaffold),sample_path=str(folded)),candidate,case)
                record.update(metrics,**confidence,refolded=True,sample_path=str(folded),backbone_path=str(scaffold))
                record['success']=design_pass(record,manifest['design_success'])
                del result
            path.write_text(json.dumps(record,indent=2));raw.append(record)
            tracker.log({method+'/raw_design/success':int(record['success']),
                method+'/raw_design/validity':int(record['validity']['valid'])},index)
    assert len(raw)==48 and len({(r['id'],r['method']) for r in raw})==48
    (output/'raw_design_scores.json').write_text(json.dumps(raw,indent=2))
    return raw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorized-continuation',action='store_true')
    args=parser.parse_args()
    assert args.authorized_continuation and os.environ.get('SLURM_JOB_ID'), 'Requires separately approved continuation budget'
    logging.basicConfig(level=logging.INFO);torch.set_num_threads(4)
    manifest=json.loads((ROOT/'benchmark_manifest.json').read_text());output=ROOT/'results'
    tracker=Tracker(output/'refold-tracking',dict(manifest_sha256=file_hash(ROOT/'benchmark_manifest.json'),source_sha256=file_hash(__file__),
        evaluator_source_sha256=file_hash(LAB/'motif_refolding.py'),sequence_count=8,mpnn_seed=2,
        criteria=manifest['design_success'],training_performed=False,raw_sequence_candidates=48,
        raw_invalid_policy='failure; no replacement, repair or exclusion'),
        'noise-matched-design-refolding',job_type='evaluation')
    original_loader=esm.pretrained.esmfold_v1;shared=[]
    def cached_loader():
        if not shared:shared.append(original_loader())
        return shared[0]
    esm.pretrained.esmfold_v1=cached_loader
    original_popen=subprocess.Popen
    def seeded_popen(args,*a,**kw):
        if isinstance(args,list) and any(str(x).endswith('/protein_mpnn_run.py') for x in args):args=args+['--seed','2']
        return original_popen(args,*a,**kw)
    subprocess.Popen=seeded_popen
    ok=False;started=time.monotonic();records=[]
    try:
        for index,row in enumerate(manifest['design_cases']):
            for method in ('native','omni'):
                case=row['case'];scaffolds=output/'scaffolds'/method/case
                dest=output/'refold'/method/case;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'scores.json').exists():
                    records.append(json.loads((dest/'scores.json').read_text()));continue
                info=dest/'motif_info.csv'
                subprocess.run([sys.executable,str(VENDOR/'scripts/write_motifInfo_from_scaffoldInfo.py'),str(scaffolds/'scaffold_info.csv'),row['reference'],str(info)],check=True)
                conf=OmegaConf.load(LAB/'config/motif_scaffolding.yaml')
                conf.inference.backbone_pdb_dir=str(scaffolds);conf.inference.output_dir=str(dest)
                conf.inference.motif_csv_path=str(info);conf.inference.motif_pdb=row['reference']
                conf.inference.samples.max_backbones=4;conf.inference.samples.seq_per_sample=8;conf.inference.samples.mpnn_batch_size=8
                conf.inference.force_motif_AA_type=True;conf.evaluation.visualize=False
                random.seed(2);np.random.seed(2);torch.manual_seed(2);torch.cuda.manual_seed_all(2)
                begin=time.monotonic();refolder=MotifRefolder(conf);refolder.run_sampling()
                complete,count=au.csv_merge(dest,'esm')
                assert count==4 and len(complete)==32,(method,case,count,len(complete))
                for column in ('rmsd','motif_rmsd','pae','plddt'):complete[column]=complete[column].astype(float)
                for i, seqrow in complete.iterrows():
                    candidate=next(r for r in manifest['design'] if r['id']==Path(seqrow['backbone_path']).stem)
                    metrics=recompute_refold_metrics(seqrow.to_dict(),candidate,row)
                    assert abs(metrics['mean_ca_distance']-float(seqrow['rmsd']))<.0006
                    complete.loc[i,'vendor_mean_ca_distance']=float(seqrow['rmsd'])
                    complete.loc[i,'rmsd']=metrics['rmsd']
                    complete.loc[i,'motif_rmsd']=metrics['motif_rmsd']
                complete['paper_success']=(complete.rmsd<=2.)&(complete.motif_rmsd<=1.)&(complete.plddt>=70.)&(complete.pae<=5.)
                complete.to_csv(dest/'sequence_scores.csv',index=False)
                scaffolds_rows=[]
                for p,g in complete.groupby('backbone_path'):
                    scaffolds_rows.append(dict(id=Path(p).stem,success=bool(g.paper_success.any()),best_sc_rmsd=float(g.rmsd.min()),best_motif_rmsd=float(g.motif_rmsd.min()),sequence_count=len(g)))
                record=dict(method=method,case=case,success=sum(r['success'] for r in scaffolds_rows),total=4,
                    seconds=time.monotonic()-begin,scaffolds=scaffolds_rows)
                (dest/'scores.json').write_text(json.dumps(record,indent=2));records.append(record)
                tracker.log({method+'/design/success_rate':record['success']/4,method+'/design/case_index':index},index)
                print('RF_BENCH_SCORED',json.dumps(record),flush=True)
        (output/'design_scores.json').write_text(json.dumps(records,indent=2))
        raw=refold_raw_sequences(manifest,output,cached_loader,tracker)
        import wandb
        artifact=wandb.Artifact('noise-matched-rfdiffusion-scores',type='evaluation')
        artifact.add_file(str(output/'design_scores.json'));artifact.add_dir(str(output/'refold'),name='refolding')
        artifact.add_file(str(output/'raw_design_scores.json'))
        artifact.add_dir(str(output/'raw_refold'),name='raw_refolding')
        tracker.run.log_artifact(artifact)
        tracker.log({'compute/process_GPU_hours':(time.monotonic()-started)/3600,'compute/peak_memory_GB':torch.cuda.max_memory_allocated()/1e9},6);ok=True
    finally:
        subprocess.Popen=original_popen;esm.pretrained.esmfold_v1=original_loader;tracker.finish(ok)


if __name__=='__main__':main()
