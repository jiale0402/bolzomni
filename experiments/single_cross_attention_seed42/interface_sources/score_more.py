"""Additional CPU-only scores of immutable Anew / frozen4M candidates."""
import concurrent.futures as cf
import hashlib,json,math,os,re,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from Bio.PDB.vectors import Vector,calc_dihedral
from common import ROOT,CONFIG,Tracker,atomic_json,sha
import sys
sys.path.insert(0,CONFIG['anew_source'])
import native_metrics as native
from residue_mapping import map_sequence_blocks_to_residues

BB=('N','CA','C','O')

def readrows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def mapped(p,row,tokens):
    target,region=native.select_coordinates(p,row,'peptide' if row['benchmark']=='lnr' else 'antibody')
    region,_=map_sequence_blocks_to_residues(tokens,region)
    return {x[0]:x for x in target},region

def atoms(res):
    return {(i,n):np.asarray(r[1][n].coord,dtype=float) for i,r in res.items() for n in BB if n in r[1]}

def aligned_rms(moving,fixed,points,truth):
    rot,trans=native.kabsch_transform(moving,fixed)
    return native.rmsd(points@rot+trans,truth)

def lddt(ref_a,gen_a,ref_b,gen_b,internal=False):
    """Backbone-only distance-preservation; missing model atoms count as failed pairs."""
    ka,kb=list(ref_a),list(ref_b)
    ra=np.array([ref_a[k] for k in ka]);rb=np.array([ref_b[k] for k in kb])
    ga=np.array([gen_a.get(k,np.full(3,np.nan)) for k in ka]);gb=np.array([gen_b.get(k,np.full(3,np.nan)) for k in kb])
    rd=cdist(ra,rb);gd=cdist(ga,gb);mask=rd<15.
    if internal:
        mask &= np.array([[x[0]!=y[0] for y in kb] for x in ka])
        mask &= np.triu(np.ones(mask.shape,dtype=bool),1)
    count=int(mask.sum())
    if not count:return None,0
    delta=np.abs(rd[mask]-gd[mask])
    return float(np.mean([np.mean(delta<t) for t in (.5,1.,2.,4.)])),count

def angle(points):
    if not all(np.isfinite(p).all() for p in points):return None
    if min(np.linalg.norm(np.cross(points[1]-points[0],points[2]-points[1])),np.linalg.norm(np.cross(points[2]-points[1],points[3]-points[2])))<1e-8:return None
    return float(np.degrees(calc_dihedral(*[Vector(p) for p in points])))

def torsions(ref,gen):
    diffs={'phi':[],'psi':[],'omega':[]};bonds=[];tested=0;missing=0
    def xyz(res,i,n):return np.asarray(res[i][1][n].coord,dtype=float)
    for i in sorted(ref):
        j=i+1
        if j not in ref or ref[i][0][0]!=ref[j][0][0]:continue
        if 'C' not in ref[i][1] or 'N' not in ref[j][1]:continue
        distance=np.linalg.norm(xyz(ref,i,'C')-xyz(ref,j,'N'))
        if distance>2.:continue  # Native chain break, not a peptide bond.
        tested+=1
        if i not in gen or j not in gen or 'C' not in gen[i][1] or 'N' not in gen[j][1]:
            missing+=1;continue
        bonds.append(abs(float(np.linalg.norm(xyz(gen,i,'C')-xyz(gen,j,'N')))-distance))
        for name,indices in {'psi':[(i,'N'),(i,'CA'),(i,'C'),(j,'N')], 'phi':[(i,'C'),(j,'N'),(j,'CA'),(j,'C')], 'omega':[(i,'CA'),(i,'C'),(j,'N'),(j,'CA')]}.items():
            if any(k not in x or n not in x[k][1] for x in (ref,gen) for k,n in indices):continue
            a,b=[angle([xyz(x,k,n) for k,n in indices]) for x in (ref,gen)]
            if a is not None and b is not None:diffs[name].append(abs((a-b+180)%360-180))
    return {**{f'{k}_mae_deg':float(np.mean(v)) if v else None for k,v in diffs.items()},'peptide_bond_length_mae':float(np.mean(bonds)) if bonds else None},{'torsions':{k:len(v) for k,v in diffs.items()},'native_connected_region_pairs':tested,'missing_generated_bonds':missing}

def protein_scores(reference,generated):
    rt,rl=reference;gt,gl=generated
    tkeys=sorted(rt.keys()&gt.keys());lkeys=sorted(rl.keys()&gl.keys())
    assert len(tkeys)>=3 and lkeys
    rat,gat,ral,gal=atoms(rt),atoms(gt),atoms(rl),atoms(gl)
    common=sorted(ral.keys()&gal.keys())
    assert len(common)>=3
    r=np.array([ral[k] for k in common]);g=np.array([gal[k] for k in common])
    rca=np.array([rt[k][2] for k in tkeys]);gca=np.array([gt[k][2] for k in tkeys])
    values={'backbone_pose_RMSD':aligned_rms(gca,rca,g,r),'backbone_internal_RMSD':aligned_rms(g,r,g,r)}
    interface,ninterface=lddt(ral,gal,rat,gat)
    internal,ninternal=lddt(ral,gal,ral,gal,True)
    values.update(backbone_interface_lddt=interface,backbone_internal_lddt=internal)
    angles,denoms=torsions(rl,gl);values.update(angles)
    denoms.update(reference_backbone_atoms=len(ral),common_backbone_atoms=len(common),interface_distance_pairs=ninterface,internal_distance_pairs=ninternal)
    return values,denoms

def seq_diversity(sequences):
    tokenized=[native.tokenize_sequence(x) for x in sequences]
    assert len({len(x) for x in tokenized})==1
    a=np.array(tokenized);dis=[]
    for i in range(len(a)):
        for j in range(i):dis.append(float(np.mean(a[i]!=a[j])))
    entropy=[]
    for col in a.T:
        _,n=np.unique(col,return_counts=True);p=n/len(col);entropy.append(float(-(p*np.log2(p)).sum()))
    return {'sequence_unique_fraction':len(set(sequences))/len(sequences),'sequence_pairwise_mismatch':float(np.mean(dis)),'sequence_position_entropy_bits':float(np.mean(entropy))}

def structural_diversity(regions):
    common=sorted(set.intersection(*[set(r) for r in regions]))
    if len(common)<3:return None,len(common),0
    arrays=[np.array([r[k][2] for k in common]) for r in regions];values=[]
    for i in range(len(arrays)):
        for j in range(i):values.append(aligned_rms(arrays[i],arrays[j],arrays[i],arrays[j]))
    return float(np.mean(values)),len(common),len(values)

def ligand_scores(pdb_sdf):
    from rdkit import Chem
    from rdkit.Chem import Descriptors,Crippen,rdMolDescriptors,FilterCatalog
    mol=Chem.SDMolSupplier(str(pdb_sdf),removeHs=True)[0]
    assert mol is not None
    values={'mol_weight':Descriptors.MolWt(mol),'cLogP':Crippen.MolLogP(mol),'TPSA':rdMolDescriptors.CalcTPSA(mol),'HBD':rdMolDescriptors.CalcNumHBD(mol),'HBA':rdMolDescriptors.CalcNumHBA(mol),'rotatable_bonds':rdMolDescriptors.CalcNumRotatableBonds(mol),'ring_count':rdMolDescriptors.CalcNumRings(mol),'fraction_CSP3':rdMolDescriptors.CalcFractionCSP3(mol),'heavy_atoms':mol.GetNumHeavyAtoms(),'single_fragment':float(len(Chem.GetMolFrags(mol))==1),'formal_charge':Chem.GetFormalCharge(mol)}
    for name in ('PAINS','BRENK'):
        if not hasattr(ligand_scores,name):
            p=FilterCatalog.FilterCatalogParams();p.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs,name));setattr(ligand_scores,name,FilterCatalog.FilterCatalog(p))
        matches=getattr(ligand_scores,name).GetMatches(mol)
        values[name.lower()+'_alert_free']=float(not matches)
    return {k:float(v) for k,v in values.items()},Chem.MolToSmiles(mol,isomericSmiles=True),mol

def target_task(task):
    model,row=task;bench,target=row['benchmark'],row['id']
    root=Path(CONFIG['models'][model]['benchmark_root']);directory=root/'scores'/bench/target
    old=readrows(directory/'evaluation.jsonl');cs={r['n']:r for r in readrows(directory/'candidate_input/results.jsonl')}
    assert len(old)==30 and {x['n'] for x in old}==set(range(30))
    result={'model':model,'benchmark':bench,'target':target,'records':[],'target_metrics':{},'target_denominators':{},'errors':{}}
    sequences=[];regions=[];smiles=[];mols=[]
    if bench!='crossdocked2020':reference=mapped(row['reference_pdb'],row,native.tokenize_sequence(row['reference_sequence']))
    for original in sorted(old,key=lambda x:x['n']):
        n=original['n'];record={'model':model,'benchmark':bench,'target':target,'n':n,'success':original['success'],'metrics':{},'errors':{},'denominators':{}}
        if original['success']:
            c=cs[n];pdb=directory/'candidate_input'/target/f'{n}.pdb'
            assert sha(pdb)==c['pdb_sha256'];record['candidate_pdb_sha256']=c['pdb_sha256']
            try:
                if bench!='crossdocked2020':
                    generated=mapped(pdb,row,native.tokenize_sequence(c['gen_seq']))
                    record['metrics'],record['denominators']=protein_scores(reference,generated)
                    sequences.append(c['gen_seq']);regions.append(generated[1])
                    if n==0:
                        identity,identity_denoms=protein_scores(reference,reference)
                        assert max(v for k,v in identity.items() if v is not None and k not in ('backbone_interface_lddt','backbone_internal_lddt'))<1e-6
                        assert all(identity[k] is None or abs(identity[k]-1.)<1e-12 for k in ('backbone_interface_lddt','backbone_internal_lddt'))
                        result['identity_self_test']=identity
                else:
                    sdf=directory/'candidate_input'/target/f'{n}.sdf'
                    record['candidate_sdf_sha256']=sha(sdf)
                    assert sha(root/'generation/omni'/bench/target/f'{n}.sdf')==record['candidate_sdf_sha256']
                    record['metrics'],s,m=ligand_scores(sdf);smiles.append(s);mols.append(m)
            except Exception:record['errors']['additional_scoring_exception']=traceback.format_exc()
        result['records'].append(record)
    if bench!='crossdocked2020' and sequences:
        result['target_metrics'].update(seq_diversity(sequences))
        diversity,nca,npair=structural_diversity(regions)
        result['target_metrics']['structure_internal_CA_diversity']=diversity
        result['target_denominators'].update(valid_sequences=len(sequences),structure_common_CA=nca,structure_pairs=npair)
    elif smiles:
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors
        fps=[rdMolDescriptors.GetMorganFingerprintAsBitVect(m,2,nBits=2048,useChirality=True) for m in mols]
        sims=[v for i,f in enumerate(fps) for v in DataStructs.BulkTanimotoSimilarity(f,fps[:i])]
        result['target_metrics'].update(canonical_smiles_unique_fraction=len(set(smiles))/len(smiles),morgan_internal_diversity=1-float(np.mean(sims)) if sims else None)
        result['target_denominators'].update(valid_molecules=len(mols),fingerprint_pairs=len(sims))
    return result

def main():
    import importlib.metadata
    from rdkit import rdBase
    started=time.monotonic();tracker=Tracker('additional-scoring',{'protocol_sha256':sha(ROOT/'code/protocol.json')})
    try:
        for name,digest in json.loads((ROOT/'DEPLOYED.json').read_text()).items():assert sha(ROOT/'code'/name)==digest,name
        tasks=[];evidence={'protocol_sha256':sha(ROOT/'code/protocol.json'),'software':{'numpy':np.__version__,'rdkit':rdBase.rdkitVersion,'scipy':importlib.metadata.version('scipy')},'models':{}}
        for model,cfg in CONFIG['models'].items():
            run=json.loads((Path(cfg['benchmark_root'])/'run.json').read_text());assert run['run_hash']==cfg['run_hash'];assert sha(run['checkpoint'])==cfg['checkpoint_sha256']
            evidence['models'][model]=cfg
            for spec in run['benchmark_specs']:
                assert sha(spec['manifest'])==spec['manifest_sha256']
                rows=readrows(spec['manifest']);assert len(rows)=={'lnr':93,'rabd':60,'crossdocked2020':100}[spec['name']]
                tasks.extend((model,r) for r in rows)
        atomic_json(ROOT/'INPUT_AUDIT.json',evidence)
        results=[];workers=int(os.environ.get('SLURM_CPUS_PER_TASK',8))
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(target_task,tasks,chunksize=1):
                results.append(result)
                errors=sum('additional_scoring_exception' in x['errors'] for x in result['records'])
                tracker.log({'targets_done':len(results),'model':result['model'],'benchmark':result['benchmark'],'target':result['target'],'valid_candidates':sum(x['success'] for x in result['records']),'new_scoring_exceptions':errors})
                atomic_json(ROOT/'PROGRESS.json',{'targets_done':len(results),'targets_expected':506,'last_target':result['target'],'updated_unix':time.time()})
        records=[x for r in results for x in r['records']]
        assert len(records)==15180
        (ROOT/'ADDITIONAL_CANDIDATES.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in records))
        atomic_json(ROOT/'TARGET_METRICS.json',[{k:v for k,v in r.items() if k!='records'} for r in results])
        summary={'targets':len(results),'fixed_attempts':len(records),'valid_candidates':sum(x['success'] for x in records),'new_scoring_exceptions':sum(bool(x['errors']) for x in records),'elapsed_seconds':time.monotonic()-started,'allocated_cpus':workers,'new_generation_attempts':0,'optimizer_updates':0,'candidate_sha256':sha(ROOT/'ADDITIONAL_CANDIDATES.jsonl'),'target_metrics_sha256':sha(ROOT/'TARGET_METRICS.json'),'protocol_sha256':sha(ROOT/'code/protocol.json')}
        atomic_json(ROOT/'SCORING_COMPLETE.json',summary)
        tracker.finish(summary,[ROOT/'ADDITIONAL_CANDIDATES.jsonl',ROOT/'TARGET_METRICS.json',ROOT/'SCORING_COMPLETE.json',ROOT/'INPUT_AUDIT.json',ROOT/'code/protocol.json'])
        assert not summary['new_scoring_exceptions'],summary
    except Exception:
        atomic_json(ROOT/'FAILED.json',{'traceback':traceback.format_exc()});tracker.run.finish(exit_code=1);raise

if __name__=='__main__':main()
