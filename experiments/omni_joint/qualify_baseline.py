"""Validation-only native folding/design qualification; not held-out comparison."""
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals
from boltzgen.task.predict.data_from_yaml import Dataset,PredictionDataset,collate
from poc_model import ROOT,ASSETS,load_model,move
from prompt import compile_prompt
from model import capture_context
from generate_panel import decode
from metrics import ca_lddt
from audit_native import rmsd


def main():
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    exp=ROOT/'experiments/omni_joint'; output=exp/'results/baseline-validation'
    output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'manifests/data.json').read_text())['validation'][:2]
    base=load_model(); canonicals=load_canonicals(str(ASSETS/'mols.zip'))
    records=[]; fasta=[]
    for row in rows:
        n=row['length']; start=n//2-8; motif=list(range(start,start+16))
        for mode in ('folding','design'):
            spec=dict(path=str(Path(row['directory'])/'native.pdb'),include=[{'chain':{'id':'A'}}],
                      structure_groups=[{'group':{'id':'A','visibility':0}}])
            if mode=='design':
                spec['design']=[{'chain':{'id':'A','res_index':','.join(str(i+1) for i in range(n) if i not in motif)}}]
                spec['structure_groups'].append({'group':{'id':'A','visibility':1,'res_index':','.join(str(i+1) for i in motif)}})
            name=row['id']+'-'+mode; path=output/(name+'.yaml')
            path.write_text(yaml.safe_dump({'entities':[{'file':spec}]},sort_keys=False))
            dataset=PredictionDataset(Dataset(str(path),Tokenizer(atomize_modified_residues=False),Featurizer()),
                canonicals,str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False)
            f=move(collate([dataset[0]]),'cuda')
            true=f['coords'].reshape(1,-1,3).clone()
            true_ca=torch.bmm(f['token_to_rep_atom'].float(),true)[0]
            bb_map=f['token_to_bb4_atoms'].float().reshape(1,n*4,-1)
            true_bb=torch.bmm(bb_map,true)[0].reshape(n,4,3)
            prompt=compile_prompt(base,f,canonical_frame=True)
            if mode=='folding':
                assert not prompt['token_distance_mask'].any()
                assert not prompt['coords'].any()
                assert not prompt['center_coords'].any()
            cache=capture_context(base,prompt)
            torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
            with torch.no_grad():
                x=base.structure_module.sample(**cache['conditioning'],atom_mask=f['atom_pad_mask'].float(),
                    multiplicity=1,num_sampling_steps=500,step_scale=2.,noise_scale=0.88)['sample_atom_coords']
            seq,ca,_=decode(x,prompt)
            bb=torch.bmm(bb_map,x)[0].reshape(n,4,3)
            record=dict(id=row['id'],mode=mode,sequence_valid='X' not in seq,sequence=seq,
                        ca_lddt=ca_lddt(ca.cpu(),true_ca.cpu()) if mode=='folding' else None,
                        ca_rmsd=rmsd(ca,true_ca) if mode=='folding' else None,
                        motif_rmsd=rmsd(bb[motif].reshape(-1,3),true_bb[motif].reshape(-1,3)) if mode=='design' else None)
            records.append(record);print(json.dumps(record),flush=True)
            np.savez_compressed(output/(name+'.npz'),backbone=bb.cpu().numpy(),native_backbone=true_bb.cpu().numpy(),motif=motif)
            if mode=='design' and 'X' not in seq:fasta.append('>'+name+'\n'+seq)
            (output/'native_metrics.json').write_text(json.dumps(records,indent=2))
    (output/'designs.fasta').write_text('\n'.join(fasta)+'\n')
    print('NATIVE_QUALIFICATION_GENERATION_COMPLETE; independent design refold still required',flush=True)


if __name__=='__main__':main()
