"""CPU: re-featurize two native structures with identical declared observations."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from Bio.PDB import PDBParser,PDBIO,Atom
from boltzgen.model.modules.masker import BoltzMasker
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals
from boltzgen.task.predict.data_from_yaml import Dataset,PredictionDataset,collate
from poc_model import ROOT,ASSETS
from prompt import compile_prompt


def main():
    torch.set_num_threads(1)
    exp=ROOT/'experiments/omni_joint'
    row=json.loads((ROOT/'manifests/data.json').read_text())['validation'][0]
    source=Path(row['directory'])
    output=exp/'results/prompt-audit'; output.mkdir(parents=True,exist_ok=True)
    structure=PDBParser(QUIET=True).get_structure('counter',source/'native.pdb')
    for index in range(row['start']+1,row['end']+2):
        residue=structure[0]['A'][(' ',index,' ')]
        residue.resname='ALA'
        for atom in list(residue):
            if atom.name not in ('N','CA','C','O'): residue.detach_child(atom.id)
        residue.add(Atom.Atom('CB',residue['CA'].coord+np.array([2.,1.,1.]),0.,1.,' ',' CB ',0,element='C'))
    writer=PDBIO(); writer.set_structure(structure); writer.save(str(output/'counter.pdb'))
    spec=yaml.safe_load((source/'fixed.yaml').read_text())
    spec['entities'][0]['file']['path']=str(output/'counter.pdb')
    counter_yaml=output/'counter.yaml'; counter_yaml.write_text(yaml.safe_dump(spec))
    canonicals=load_canonicals(str(ASSETS/'mols.zip'))
    base=SimpleNamespace(masker=BoltzMasker(mask=True,mask_backbone=False))
    def compile(path):
        random.seed(1729); np.random.seed(1729); torch.manual_seed(1729)
        dataset=PredictionDataset(Dataset(str(path),Tokenizer(atomize_modified_residues=False),Featurizer()),
            canonicals,str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False)
        default_rng=np.random.default_rng
        np.random.default_rng=lambda value=None:default_rng(1729 if value is None else value)
        try: f=collate([dataset[0]])
        finally: np.random.default_rng=default_rng
        return compile_prompt(base,f,canonical_frame=True)
    a,b=compile(source/'fixed.yaml'),compile(counter_yaml)
    differences={}
    for key,x in a.items():
        if not isinstance(x,torch.Tensor): continue
        y=b[key]
        if x.shape!=y.shape: differences[key]={'shape_a':list(x.shape),'shape_b':list(y.shape)}
        elif not torch.equal(x,y):
            differences[key]={'max_difference':float((x.double()-y.double()).abs().max())}
    # Emit all differences, not an assumed pass. Downstream qualification must classify
    # metadata vs actual neural inputs and resolve any hidden-information dependence.
    (output/'differences.json').write_text(json.dumps(differences,indent=2))
    print('REFEATURIZED_COUNTERFACTUAL_DIFFERENCES',json.dumps(differences),flush=True)
    assert not differences, 'Recompiled prompts must not depend on hidden residue identity'


if __name__=='__main__': main()
