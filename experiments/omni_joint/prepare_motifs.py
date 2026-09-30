"""Compile published contigs into paired, explicitly masked BoltzGen prompts."""
import copy
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
import numpy as np
import yaml
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom

EXP=Path(__file__).resolve().parent


def numeric(part):
    fields=list(map(int,part.split('-')))
    return fields[0],fields[-1]


def parse_segment(segment):
    m=re.fullmatch(r'([A-Za-z])([0-9]+)(?:-([0-9]+))?',segment)
    if not m: raise ValueError(segment)
    return m[1],int(m[2]),int(m[3] or m[2])


def choose_layout(task,seed):
    rng=random.Random(seed)
    segments=task['contig'].split('/')
    motif_length=sum(parse_segment(s)[2]-parse_segment(s)[1]+1 for s in segments if s[0].isalpha())
    for _ in range(100000):
        gaps=[rng.randint(*numeric(s)) for s in segments if not s[0].isalpha()]
        if task['length'][0]<=motif_length+sum(gaps)<=task['length'][1]:
            return gaps
    raise RuntimeError('No valid layout under published constraints: '+task['id'])


def main():
    specification=json.loads((EXP/'motif_tasks.json').read_text())
    tasks=specification['tasks']; assert len(tasks)==25
    root=EXP/'motif_data'; root.mkdir(exist_ok=True)
    manifest=[]
    for task in tasks:
        source=root/(task['pdb']+'.pdb')
        source_url=f'https://files.rcsb.org/download/{task["pdb"]}.pdb'
        if task['pdb']=='6EXZ':
            # Published motif A28-42 uses the renumbered chain (PDB author IDs
            # 557-571). Use a pinned public benchmark reference, not guessed IDs.
            source=EXP/'6exz_reference.pdb'
            source_url='https://raw.githubusercontent.com/microsoft/evodiff/33206e99446f799ec11cf6e57d66ffaec837be91/examples/scaffolding-pdbs/6exz_reres.pdb'
        if not source.exists():
            with urllib.request.urlopen(source_url,timeout=45) as response:
                source.with_suffix('.tmp').write_bytes(response.read())
            source.with_suffix('.tmp').replace(source)
        source_model=PDBParser(QUIET=True).get_structure(task['pdb'],source)[0]
        for candidate in range(16):
            seed=int.from_bytes(hashlib.sha256(f'omni-joint:{task["id"]}:{candidate}'.encode()).digest()[:4],'little')
            gaps=iter(choose_layout(task,seed))
            structure=Structure.Structure('prompt'); model=Model.Model(0); structure.add(model)
            chain=Chain.Chain('A'); model.add(chain)
            motif=[]; generated=[]; mapping=[]
            def append(residue,origin=None):
                rid=len(chain)+1
                residue=residue.copy(); residue.detach_parent(); residue.id=(' ',rid,' ')
                chain.add(residue)
                if origin:
                    motif.append(rid); mapping.append(dict(output_chain='A',output_residue=rid,
                                                        native_chain=origin[0],native_residue=origin[1]))
                else: generated.append(rid)
            for segment in task['contig'].split('/'):
                if segment[0].isalpha():
                    cid,start,end=parse_segment(segment)
                    for index in range(start,end+1):
                        residue=source_model[cid][(' ',index,' ')]
                        assert all(a in residue for a in ('N','CA','C','O')),(task['id'],cid,index)
                        append(residue,(cid,index))
                else:
                    for _ in range(next(gaps)):
                        residue=Residue.Residue((' ',1,' '),'ALA',' ')
                        for serial,name in enumerate(('N','CA','C','O','CB'),1):
                            residue.add(Atom.Atom(name,np.zeros(3),0.,1.,' ',f'{name:>4}',serial,element=name[0]))
                        append(residue)
            assert task['length'][0]<=len(chain)<=task['length'][1]
            if 'target' in task:
                target_chain=Chain.Chain('B'); model.add(target_chain)
                cid,start,end=parse_segment(task['target'])
                for index in range(start,end+1):
                    residue=source_model[cid][(' ',index,' ')].copy()
                    residue.detach_parent(); residue.id=(' ',len(target_chain)+1,' '); target_chain.add(residue)
            directory=root/f'{task["id"]}-candidate-{candidate:02d}'
            directory.mkdir(exist_ok=True)
            pdb=directory/'prompt.pdb'; writer=PDBIO(); writer.set_structure(structure); writer.save(str(pdb))
            spec=dict(path=str(pdb),include=[{'chain':{'id':'A'}}],
                design=[{'chain':{'id':'A','res_index':','.join(map(str,generated))}}],
                structure_groups=[{'group':{'id':'A','visibility':0}},
                                  {'group':{'id':'A','visibility':1,'res_index':','.join(map(str,motif))}}])
            if 'target' in task:
                spec['include'].append({'chain':{'id':'B'}})
                spec['structure_groups'].append({'group':{'id':'B','visibility':1}})
            yaml_path=directory/'design.yaml'
            yaml_path.write_text(yaml.safe_dump({'entities':[{'file':spec}]},sort_keys=False))
            manifest.append(dict(task=task['id'],candidate=candidate,seed=seed,yaml=str(yaml_path),
                                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_url=source_url,
                                 design_length=len(chain),motif_mapping=mapping,generated_residues=generated))
        print('MOTIF_PREPARED',task['id'],flush=True)
    (EXP/'manifests').mkdir(exist_ok=True)
    (EXP/'manifests/motifs.json').write_text(json.dumps(manifest,indent=2))
    print('MOTIF_MANIFEST_COMPLETE',len(manifest),flush=True)


if __name__=='__main__':main()
