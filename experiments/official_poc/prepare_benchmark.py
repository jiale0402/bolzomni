"""Lock an author-released folding panel and Table 8 RFdiffusion design tasks."""
import json
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from tracking import file_hash

ROOT=Path(__file__).resolve().parent
TASKS=['6E6R-short','1YCR','1PRW','5YUI','5IUS','1BCF']


def main():
    path=ROOT/'benchmark_manifest.json'
    if path.exists():return
    data=json.loads((ROOT/'data_manifest.json').read_text())
    old=json.loads((ROOT.parent/'omni_joint/manifests/motifs.json').read_text())
    root=ROOT/'benchmark';root.mkdir(exist_ok=True)
    rows=[];cases=[]
    for task in TASKS:
        selected=[r for r in old if r['task']==task and r['candidate']<4]
        assert len(selected)==4
        reference=root/(task+'.pdb');reference_lines=[];infos=[]
        for row in selected:
            prompt=Path(row['yaml']).with_name('prompt.pdb')
            atoms=PDBParser(QUIET=True).get_structure('s',prompt)[0]['A']
            mapping=row['motif_mapping'];segments=[]
            for m in mapping:
                if not segments or m['output_residue']!=segments[-1][-1]['output_residue']+1 or m['native_chain']!=segments[-1][-1]['native_chain'] or m['native_residue']!=segments[-1][-1]['native_residue']+1:
                    segments.append([])
                segments[-1].append(m)
            parts=[];cursor=1;motif_indices=[]
            for i,segment in enumerate(segments):
                first=segment[0]['output_residue'];last=segment[-1]['output_residue'];letter=chr(65+i)
                if first>cursor:parts.append(str(first-cursor))
                parts.append(letter);cursor=last+1
                motif_indices.extend(m['output_residue'] for m in segment)
                if row['candidate']==0:
                    for j,m in enumerate(segment,1):
                        residue=atoms[(' ',m['output_residue'],' ')]
                        for atom in residue:
                            pos=atom.coord;name=atom.name;serial=len(reference_lines)+1
                            reference_lines.append(f'ATOM  {serial:5d} {name:^4s} {residue.resname:>3s} {letter}{j:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{1.:6.2f}{0.:6.2f}          {name[0]:>2s}  ')
            if cursor<=row['design_length']:parts.append(str(row['design_length']-cursor+1))
            rows.append(dict(row,id=f'{task}_{row["candidate"]}',case=task,length=row['design_length'],group=len(segments),motif_placements='/'.join(parts),motif_indices=motif_indices,
                input_sha256=file_hash(row['yaml']),prompt_sha256=file_hash(prompt)))
            infos.append(dict(sample_num=row['candidate'],motif_placements='/'.join(parts)))
        reference.write_text('\n'.join(reference_lines)+'\nTER\nEND\n')
        cases.append(dict(case=task,reference=str(reference),reference_sha256=file_hash(reference),scaffold_info=infos))
    manifest=dict(folding=data['test'],design=rows,design_cases=cases,
        folding_source=data['folding_panel_source'],folding_source_sha256=data['folding_panel_source_sha256'],
        folding_limitation='Official released monomer validation panel, not the unavailable 2024 Figure 15 test manifest; sequence-only, no MSA.',
        design_source='BoltzGen paper Appendix C.5 / Table 8; published RFdiffusion contigs',
        folding_samples=5,design_samples=4,sampling_steps=500,recycling_steps=3,mpnn_sequences=8,
        design_success={'ca_rmsd_max':2.,'motif_rmsd_max':1.,'plddt_min':70.,'pae_max':5.},
        checkpoint='experiments/official_poc/runs/corrected-1729/final.pt',
        selection='Six fixed tasks covering 1,2,3,4 motif segments; four pre-existing fixed layouts/seeds per task; selected before model evaluation.')
    path.write_text(json.dumps(manifest,indent=2))
    print('BENCHMARK_LOCKED',len(manifest['folding']),len(rows),file_hash(path),flush=True)


if __name__=='__main__':main()
