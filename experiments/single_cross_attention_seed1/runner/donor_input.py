"""Build native Boltz protein slots and visible context without generated identity.

Nonprotein context lacking a validated block-to-token map is explicitly uncovered.
Unknown small-molecule generated blocks never enter the donor input.
"""
import gzip
import json
import math
from pathlib import Path
import gemmi
import yaml

AA = set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())


def prepare(root, key):
    rec = json.loads(gzip.decompress((root/'records'/(key+'.json.gz')).read_bytes()))
    dest = root/'inputs'/key
    selected = {(c,tuple(r)):bool(g) for (c,r),g in zip(rec['select_indexes'],rec['generate_mask'])}
    segments, lookup, excluded = [], {}, []
    for chain in rec['target_chains']:
        oldchain = chain[2]
        grouped, aliases = {}, {}
        for name, atoms, rid, props in chain[1]:
            actual = props.get('original_name',name)
            realid = (rid[0], ''.join(c for c in rid[1] if not c.isdigit())) if 'original_name' in props else tuple(rid)
            groupkey = (realid, actual)
            if groupkey not in grouped:
                grouped[groupkey] = [actual, [], list(realid), {}]
                aliases[groupkey] = []
            grouped[groupkey][1].extend(atoms)
            aliases[groupkey].append(tuple(rid))
        prev, segment = None, None
        for groupkey, block in grouped.items():
            name, atoms, rid, props = block
            gens = {selected[(oldchain, alias)] for alias in aliases[groupkey]}
            assert len(gens)==1
            gen = gens.pop()
            if name not in AA and not {'N','CA','C'}<=set(a[0] for a in atoms):
                for alias in aliases[groupkey]:lookup[(oldchain,alias)] = None
                excluded.append({'chain':oldchain,'residue':rid,'name':name,'reason':'no audited protein-residue correspondence'})
                prev=None
                continue
            assert len({a[0] for a in atoms}) == len(atoms), (key,name,rid)
            coords = {a[0]:a[1] for a in atoms}
            linked = False
            if prev:
                pb, pg = prev
                pc = {a[0]:a[1] for a in pb[1]}
                adjacent = rid[0] == pb[2][0]+1 or (rid[0]==pb[2][0] and rid[1]!=pb[2][1])
                linked = adjacent and (gen or pg or ('C' in pc and 'N' in coords and math.dist(pc['C'],coords['N'])<=2.0))
            if not linked:
                segment=[]
                segments.append(segment)
            cid='T'+str(len(segments)-1)
            idx=len(segment)
            segment.append((block,gen))
            for alias in aliases[groupkey]:lookup[(oldchain,alias)]=[cid,idx]
            prev=(block,gen)
    mapping=[]
    for (chain,rid),gen in zip(rec['select_indexes'],rec['generate_mask']):
        if gen and rec['modality']=='ligand':mapping.append(None)
        else:mapping.append(lookup.get((chain,tuple(rid))))
    meta={'key':key,'mapping':mapping,'generate_mask':rec['generate_mask'],'modality':rec['modality'],
        'excluded_context':excluded,'source_record_sha256':rec['raw_record_sha256'],
        'donor_input_scope':'native selected context; native antibody CDR slots; no unknown molecule slots',
        'native_generated_identity_exported':False}
    if not segments:
        meta['empty_donor']=True
        (dest/'metadata.json').write_text(json.dumps(meta,indent=2))
        return None,meta
    doc=gemmi.cif.Document();b=doc.add_new_block(key)
    # Gemmi block edits can invalidate previously returned Loop handles.
    # Reacquire each loop by its first tag when adding a row.
    class StableLoop:
        def __init__(self, tag):self.tag=tag
        def add_row(self, row):b.find_loop(self.tag).get_loop().add_row(row)
    def loop(prefix,cols):
        names=cols.split()
        b.init_loop(prefix,names)
        return StableLoop(prefix+names[0])
    ent=loop('_entity.','id type');poly=loop('_entity_poly.','entity_id type');asym=loop('_struct_asym.','id entity_id')
    seq=loop('_entity_poly_seq.','entity_id num mon_id hetero');comp=loop('_chem_comp.','id type')
    at=loop('_atom_site.','group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv auth_seq_id auth_comp_id auth_asym_id pdbx_PDB_model_num')
    for name in sorted({r[0][0] for s in segments for r in s}):comp.add_row([name,"'L-peptide linking'"])
    design, visibility, serial = [], [], 0
    for i,segment in enumerate(segments):
        eid=str(i+1);cid='T'+str(i)
        ent.add_row([eid,'polymer']);poly.add_row([eid,"'polypeptide(L)'"]);asym.add_row([cid,eid])
        visibility.append({'group':{'id':cid,'visibility':1}})
        gen_indices=[]
        for j,(block,gen) in enumerate(segment,1):
            name,atoms,rid,props=block
            if gen:gen_indices.append(j)
            seq.add_row([eid,str(j),name,'n'])
            for an,xyz,el,aid,ap in atoms:
                if el in ['H','D']:continue
                assert all(math.isfinite(v) for v in xyz)
                if gen:assert name=='GLY' and all(v==0 for v in xyz)
                serial+=1
                at.add_row(['ATOM',str(serial),el,gemmi.cif.quote(an),'.',name,cid,eid,str(j),'?']+
                    [f'{v:.6f}' for v in xyz]+['1.0','0.0',str(j),name,cid,'1'])
        if gen_indices:
            indices=','.join(map(str,gen_indices))
            design.append({'chain':{'id':cid,'res_index':indices}})
            visibility.append({'group':{'id':cid,'res_index':indices,'visibility':0}})
    source=dest/'target.cif';doc.write_file(str(source))
    file_spec={'path':str(source),'include':[{'chain':{'id':'T'+str(i)}} for i in range(len(segments))],
        'structure_groups':visibility}
    if design:file_spec['design']=design
    path=dest/'task.yaml';path.write_text(yaml.safe_dump({'entities':[{'file':file_spec}]},sort_keys=False))
    (dest/'metadata.json').write_text(json.dumps(meta,indent=2))
    return path,meta
