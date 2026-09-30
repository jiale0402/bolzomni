"""Remove adaptation homologs of held-out chains and benchmark source chains."""
import hashlib
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

EXP=Path(__file__).resolve().parent


def main():
    work=EXP/'overlap'; work.mkdir(exist_ok=True)
    tool=work/'mmseqs/bin/mmseqs'
    url='https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-avx2.tar.gz'
    archive=work/'mmseqs-18-8cc5c.tar.gz'
    if not tool.exists():
        with urllib.request.urlopen(url,timeout=60) as response: archive.write_bytes(response.read())
        with tarfile.open(archive) as tar: tar.extractall(work,filter='data')
    rows=json.loads((EXP/'manifests/data.json').read_text())
    tasks=json.loads((EXP/'motif_tasks.json').read_text())['tasks']
    protected={r['id']:r['sequence'] for r in rows['test']}
    parser=PDBParser(QUIET=True)
    for task in tasks:
        structure=parser.get_structure(task['pdb'],EXP/'motif_data'/(task['pdb']+'.pdb'))[0]
        chains={s[0] for s in task['contig'].split('/') if s[0].isalpha()}
        if 'target' in task: chains.add(task['target'][0])
        for chain in chains:
            seq=''.join(seq1(r.resname,undef_code='X') for r in structure[chain] if r.id[0]==' ')
            protected['motif-'+task['pdb']+'-'+chain]=seq
    exclusions=[]
    def fasta(path,records):
        path.write_text(''.join('>'+name+'\n'+seq+'\n' for name,seq in records.items()))
    for split in ('validation','train'):
        query=work/(split+'.fasta'); target=work/(split+'-protected.fasta')
        fasta(query,{r['id']:r['sequence'] for r in rows[split]}); fasta(target,protected)
        hits=work/(split+'-hits.tsv')
        command=[str(tool),'easy-search',str(query),str(target),str(hits),str(work/(split+'-tmp')),
                 '--min-seq-id','0.3','-c','0.8','--cov-mode','2','-s','7.5',
                 '--alignment-mode','3','--format-output','query,target,fident,qcov,tcov','--threads','4']
        subprocess.run(command,check=True,stdout=open(work/(split+'-mmseqs.log'),'w'),stderr=subprocess.STDOUT)
        excluded=set()
        for line in hits.read_text().splitlines():
            query_id,target_id,identity,qcov,tcov=line.split('\t')
            excluded.add(query_id)
            exclusions.append(dict(split=split,id=query_id,reference=target_id,identity=float(identity),query_coverage=float(qcov),target_coverage=float(tcov)))
        rows[split]=[r for r in rows[split] if r['id'] not in excluded]
        protected.update({r['id']:r['sequence'] for r in rows[split]})
    report=dict(counts={k:len(v) for k,v in rows.items()},mmseqs_version='18-8cc5c',
                executable_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
                minimum_identity=0.3,minimum_query_coverage=0.8,exclusions=exclusions)
    (EXP/'manifests/data_filtered.json').write_text(json.dumps(rows,indent=2))
    (EXP/'manifests/overlap_audit.json').write_text(json.dumps(report,indent=2))
    print('OVERLAP_FILTER_COMPLETE',json.dumps(report['counts']),len(exclusions),flush=True)


if __name__=='__main__':main()
