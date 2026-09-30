"""CPU input audit, without weights or predictions: visibility and reference mapping."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.task.predict.data_from_yaml import Dataset,PredictionDataset,collate
from boltzgen.model.modules.masker import BoltzMasker
from poc_model import ASSETS
from prompt import compile_prompt
from tracking import file_hash

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
manifest=json.loads((ROOT/'manifest.json').read_text())
canonicals=load_canonicals(str(ASSETS/'mols.zip'))
base=SimpleNamespace(masker=BoltzMasker(mask=True,mask_backbone=False))
records=[]
for row in manifest['folding']:
    count=0
    for line in Path(row['reference']).read_text().splitlines():
        if line.startswith('ATOM') and line[12:16].strip()=='CA' and line[16] in ' A':
            pos=int(line[22:26]);letter=const.prot_token_to_letter[line[17:20].strip()]
            assert letter==row['sequence'][pos-1],(row['id'],pos,letter,row['sequence'][pos-1])
            count+=1
    assert count==len(row['resolved_ca'])
    records.append(dict(id=row['id'],reference_sequence_mapping=True,resolved_ca=count))
for row in manifest['design'][::2]:
    dataset=PredictionDataset(Dataset(row['yaml'],Tokenizer(atomize_modified_residues=False),Featurizer()),
        canonicals,str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False)
    f=collate([dataset[0]])
    prompt=compile_prompt(base,f,canonical_frame=True)
    visible=prompt['token_distance_mask'].bool().any(-1)[0]
    known=~prompt['design_mask'].bool()[0]
    assert (visible.nonzero().flatten()+1).tolist()==row['motif'],row['id']
    assert (known.nonzero().flatten()+1).tolist()==row['fixed_identity'],row['id']
    atomvisible=torch.bmm(prompt['atom_to_token'].float(),visible.float()[None,:,None]).squeeze(-1).bool()
    hard=atomvisible & prompt['backbone_mask'].bool() & prompt['atom_pad_mask'].bool()
    xyz=prompt['coords'].reshape(1,-1,3)
    assert not xyz[~hard].any()
    motif_ca=torch.bmm(prompt['token_to_rep_atom'].float(),xyz)[0,visible].numpy()
    expected=np.array([[float(l[a:b]) for a,b in ((30,38),(38,46),(46,54))]
        for l in Path(row['reference']).read_text().splitlines()
        if l.startswith('ATOM') and l[12:16].strip()=='CA'])
    assert motif_ca.shape==expected.shape
    assert np.max(abs(motif_ca-expected))<.002,(row['id'],np.max(abs(motif_ca-expected)))
    records.append(dict(id=row['case'],coordinate_visibility_exact=True,identity_visibility_exact=True,
        hidden_coordinates_zero=True,motif_coordinates_match_reference=True,observed_backbone_atoms=int(hard.sum())))
(ROOT/'input_audit.json').write_text(json.dumps(dict(passed=True,
    manifest_sha256=file_hash(ROOT/'manifest.json'),source_sha256=file_hash(__file__),records=records),indent=2))
print('ALL_BENCHMARK_INPUTS_AUDITED',len(records),flush=True)
