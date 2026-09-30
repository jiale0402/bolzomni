"""Lock a small stratified public benchmark subset before running either model."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
CASP=ROOT/'data/casp16'
MOTIF=ROOT/'data/motifbench'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rank(x):return hashlib.sha256(('standard-bench-v1:'+x).encode()).hexdigest()

def main():
    import numpy as np
    import yaml
    from Bio.PDB import PDBParser,PDBIO,Structure,Model,Chain,Residue,Atom
    seqs={}
    for line in (CASP/'sequences.fasta').read_text().splitlines():
        if line.startswith('>'):key=line[1:].split()[0];seqs[key]=''
        else:seqs[key]+=line.strip()
    eligible=[]
    for key,sequence in seqs.items():
        path=CASP/'targets/Targets_processed'/f'{key}.pdb'
        if not (80<=len(sequence)<=400 and path.exists() and set(sequence)<=set('ACDEFGHIKLMNPQRSTVWY')):continue
        ca={int(l[22:26]):[float(l[a:b]) for a,b in ((30,38),(38,46),(46,54))]
            for l in path.read_text().splitlines() if l.startswith('ATOM') and l[12:16].strip()=='CA' and l[16] in ' A'}
        if len(ca)/len(sequence)<.7 or min(ca)<1 or max(ca)>len(sequence):continue
        eligible.append(dict(id=key,sequence=sequence,length=len(sequence),reference=str(path),
                             reference_sha256=digest(path),resolved_ca=ca,coverage=len(ca)/len(sequence)))
    # Screen adaptation train/validation overlap using the existing audited MMseqs settings.
    data=json.loads((PROJECT/'experiments/omni_joint/manifests/data_filtered.json').read_text())
    overlap=ROOT/'overlap';overlap.mkdir(exist_ok=True)
    (overlap/'query.fasta').write_text(''.join('>'+r['id']+'\n'+r['sequence']+'\n' for r in eligible))
    (overlap/'adaptation.fasta').write_text(''.join('>'+r['id']+'\n'+r['sequence']+'\n' for s in ('train','validation') for r in data[s]))
    mmseqs=PROJECT/'experiments/omni_joint/overlap/mmseqs/bin/mmseqs'
    with open(overlap/'search.log','w') as log:
        subprocess.run([str(mmseqs),'easy-search',str(overlap/'query.fasta'),str(overlap/'adaptation.fasta'),
            str(overlap/'hits.tsv'),str(overlap/'tmp'),'--min-seq-id','0.3','-c','0.8','--cov-mode','2',
            '-s','7.5','--alignment-mode','3','--format-output','query,target','--threads','4'],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    excluded={l.split('\t')[0] for l in (overlap/'hits.tsv').read_text().splitlines()}
    folded=[];families=set()
    for low,high,count in ((80,150,3),(151,250,4),(251,400,3)):
        choices=sorted([r for r in eligible if low<=r['length']<=high and r['id'] not in excluded],key=lambda r:rank(r['id']))
        chosen=[]
        for row in choices:
            family=re.match(r'T\d+',row['id'])[0]
            if family in families:continue
            chosen.append(row);families.add(family)
            if len(chosen)==count:break
        assert len(chosen)==count,(low,high,count,len(chosen))
        folded.extend(chosen)
    cases=list(csv.DictReader((MOTIF/'test_cases.csv').open()))
    for i,row in enumerate(cases,1):
        row.update(case=f'{i:02d}_{row["pdb_id"]}',length=int(row['length']),group=int(row['group']))
    chosen=[]
    for group in (1,2,3):
        pool=[r for r in cases if r['group']==group]
        short=sorted(pool,key=lambda r:(r['length'],rank(r['case'])))[0]
        long=sorted([r for r in pool if r['pdb_id']!=short['pdb_id']],key=lambda r:(-r['length'],rank(r['case'])))[0]
        chosen.extend([short,long])
    prompts=[];parser=PDBParser(QUIET=True)
    for row in chosen:
        source=MOTIF/'motif_pdbs'/(row['case']+'.pdb')
        model=parser.get_structure(row['case'],source)[0]
        segments=[list(c.get_residues()) for c in model]
        free=row['length']-sum(map(len,segments))
        assert free>=0
        gaps=[free//(len(segments)+1)]*(len(segments)+1)
        for i in range(free%len(gaps)):gaps[i]+=1
        structure=Structure.Structure('prompt');m=Model.Model(0);structure.add(m);chain=Chain.Chain('A');m.add(chain)
        motif=[];fixed=[];design=[];mapping=[];placement=[]
        def append(residue,segment=None,original=None):
            rid=len(chain)+1;r=residue.copy();r.detach_parent();r.id=(' ',rid,' ')
            unknown=r.resname=='UNK'
            if unknown:r.resname='ALA'
            chain.add(r)
            if segment is not None:
                motif.append(rid);mapping.append(dict(output_residue=rid,chain=segment,reference_residue=original))
                (design if unknown else fixed).append(rid)
            else:design.append(rid)
        for i,gap in enumerate(gaps):
            if gap:placement.append(str(gap))
            for j in range(gap):
                r=Residue.Residue((' ',1,' '),'ALA',' ')
                for serial,name in enumerate(('N','CA','C','O','CB'),1):
                    r.add(Atom.Atom(name,np.zeros(3),0.,1.,' ',f'{name:>4}',serial,element=name[0]))
                append(r)
            if i<len(segments):
                cid=segments[i][0].parent.id;placement.append(cid)
                for r in segments[i]:append(r,cid,r.id[1])
        assert len(chain)==row['length']
        directory=ROOT/'prompts'/row['case'];directory.mkdir(parents=True,exist_ok=True)
        pdb=directory/'prompt.pdb';writer=PDBIO();writer.set_structure(structure);writer.save(str(pdb))
        spec=dict(path=str(pdb),include=[{'chain':{'id':'A'}}],
            design=[{'chain':{'id':'A','res_index':','.join(map(str,design))}}],
            structure_groups=[{'group':{'id':'A','visibility':0}},
                {'group':{'id':'A','visibility':1,'res_index':','.join(map(str,motif))}}])
        yaml_path=directory/'design.yaml';yaml_path.write_text(yaml.safe_dump({'entities':[{'file':spec}]},sort_keys=False))
        for candidate in range(2):
            seed=int(rank(row['case']+str(candidate))[:8],16)%(2**31)
            prompts.append(dict(**row,candidate=candidate,id=f'{row["case"]}_{candidate}',seed=seed,yaml=str(yaml_path),
                reference=str(source),reference_sha256=digest(source),motif=motif,fixed_identity=fixed,
                motif_mapping=mapping,motif_placements='/'.join(placement)))
    manifest=dict(folding=folded,design=prompts,motif_cases=chosen,
        selection='CASP16: 3 short/4 medium/3 long, >=70% resolved CA, unique target families, hash order; MotifBench: shortest and longest distinct-PDB case per motif-segment group',
        adaptation_overlap_excluded=sorted(excluded),motifbench_revision='d5a714c6a376c701e29dfa66d42a1b13c5e339e0',
        folding_metrics=['biotite lDDT on resolved CA atoms','TM-score'],
        design_metric='Official MotifBench ProteinMPNN(8 sequences)+ESMFold refolding success (scRMSD<2A, motifRMSD<1A)',
        design_scope='Scaffold capability under official sequence redesign; generated Omni sequence consistency is a separate diagnostic, not a gate.',
        full_leaderboard=False,checkpoint='experiments/omni_joint/runs/full-poc-seed-1729/final.pt',
        base_pretraining_exposure='not excluded; retrospective paired benchmark, not blind generalization')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('BENCHMARK_LOCKED',[(r['id'],r['length']) for r in folded],[(r['case'],r['group'],r['length']) for r in chosen],flush=True)

if __name__=='__main__':main()
