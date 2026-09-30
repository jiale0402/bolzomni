"""Use actual NPZ members in the official archive; retain bytes and provenance."""
import concurrent.futures
import hashlib
import io
import json
import struct
import zlib
import urllib.request
from pathlib import Path
import numpy as np
from remote_zip import open_archive, RangeFile, REVISION

ROOT = Path(__file__).resolve().parent
AA = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), 'ARNDCQEGHILKMFPSTWYV'))


def member(info):
    path = ROOT/'data'/info.filename
    if path.exists(): return path
    source = RangeFile(ROOT/'data/ranges')
    source.seek(info.header_offset)
    raw = source.read(30+len(info.filename.encode())+256+info.compress_size)
    header = struct.unpack_from('<4s5H3I2H', raw)
    assert header[0] == b'PK\x03\x04'
    start = 30+header[-2]+header[-1]
    compressed = raw[start:start+info.compress_size]
    data = zlib.decompress(compressed, -15) if info.compress_type == 8 else compressed
    assert len(data) == info.file_size and zlib.crc32(data) == info.CRC
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return path


def convert(row, info, require_complete=True):
    raw = member(info)
    with np.load(raw) as npz:
        chains, atoms, residues = npz['chains'], npz['atoms'], npz['residues']
        assert len(chains) == 1 and chains[0]['mol_type'] == 0
        coords, sequence, lines, resolved_ca = [], [], [], {}
        serial = 0
        for rid, residue in enumerate(residues, 1):
            name = str(residue['name'])
            if name not in AA: return None
            sequence.append(AA[name])
            aa = atoms[int(residue['atom_idx']):int(residue['atom_idx']+residue['atom_num'])]
            present = {str(a['name']):a for a in aa if a['is_present']}
            if require_complete and not all(n in present for n in ('N','CA','C','O')): return None
            if 'CA' in present:
                coords.append(present['CA']['coords'].tolist())
                resolved_ca[str(rid)] = present['CA']['coords'].tolist()
            for atom_name, atom in present.items():
                pos = atom['coords']; serial += 1
                lines.append(f'ATOM  {serial:5d} {atom_name:^4s} {name:>3s} A{rid:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{1.:6.2f}{float(atom["bfactor"]):6.2f}          {atom_name[0]:>2s}  ')
        distances = np.linalg.norm(np.diff(np.array(coords), axis=0), axis=-1)
        if not np.isfinite(coords).all() or len(coords)<.7*len(sequence): return None
        if require_complete and np.any((distances<3.4)|(distances>4.2)): return None
        dest = ROOT/'data/prepared'/row['id'];dest.mkdir(parents=True, exist_ok=True)
        pdb = dest/'native.pdb';pdb.write_text('\n'.join(lines)+'\nTER\nEND\n')
        (dest/'official_record.json').write_text(json.dumps(row, indent=2))
        return dict(id=row['id'], sequence=''.join(sequence), length=len(sequence), directory=str(dest),
            source='boltzgen/boltzgen1_train', dataset_revision=REVISION, archive_member=info.filename,
            member_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), pdb_sha256=hashlib.sha256(pdb.read_bytes()).hexdigest(),
            official_record=row, resolved_ca=resolved_ca)


def main():
    source, archive = open_archive(ROOT/'data/ranges')
    candidates = json.loads((ROOT/'data/candidates.json').read_text())
    split = ROOT/'data/val_monomers_boltzgen_min50_max220.txt'
    if not split.exists():
        split.write_bytes(urllib.request.urlopen('https://raw.githubusercontent.com/HannesStark/boltzgen/a3149cf18eeb58648d1abbb27539bd73f746cdda/src/boltzgen/resources/splits/val_monomers_boltzgen_min50_max220.txt').read())
    heldout_ids = {s.strip().lower() for s in split.read_text().splitlines()}
    names = set(archive.namelist())
    def read_record(p):
        name = f'targets/records/{p}.json'
        return json.loads(member(archive.getinfo(name)).read_text()) if name in names else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        official_val = list(pool.map(read_record, sorted(heldout_ids)))
    candidates['testing'] = sorted([r for r in official_val if r and len(r['chains'])==1 and r['chains'][0]['mol_type']==0 and r['chains'][0]['valid']], key=lambda r:hashlib.sha256(('1729:'+r['id']).encode()).hexdigest())
    candidates['training'] = [r for r in candidates['training'] if r['id'].lower() not in heldout_ids]
    all_motif_ids = {t['pdb'].lower() for t in json.loads((ROOT.parent/'omni_joint/motif_tasks.json').read_text())['tasks']}
    result = dict(train=[], validation=[], test=[], rejections=[])
    seen_clusters, seen_sequences = set(), set()
    for kind, rows, wanted in [('test', candidates['testing'], 8), ('train', candidates['training'], 144)]:
        for start in range(0, len(rows), 16):
            batch = [r for r in rows[start:start+16] if r['id'].lower() not in all_motif_ids]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(convert, r, archive.getinfo(f'targets/structures/{r["id"]}.npz'), kind!='test') for r in batch]
                converted = [f.result() for f in futures]
            for row, item in zip(batch, converted):
                cluster = row['chains'][0]['cluster_id']
                if item is None:
                    result['rejections'].append(dict(id=row['id'], reason='noncanonical/incomplete backbone/chain break'));continue
                if cluster in seen_clusters or item['sequence'] in seen_sequences: continue
                seen_clusters.add(cluster);seen_sequences.add(item['sequence'])
                result[kind].append(item)
                if len(result[kind]) == wanted: break
            print('PREPARED', kind, len(result[kind]), flush=True)
            if len(result[kind]) == wanted: break
        assert len(result[kind]) == wanted, (kind,len(result[kind]),wanted)
    result['validation'] = result['train'][-16:];result['train'] = result['train'][:128]
    result['provenance'] = {k:candidates[k] for k in ('dataset_revision','archive_url','split_sha256','rules')}
    result['selection'] = 'SHA256(1729:PDB), unique official sequence clusters, 128 train/16 validation/8 released BoltzGen monomer-validation structures. The released validation list is NOT claimed to be the unavailable Figure 15 test manifest.'
    result['folding_panel_source'] = 'src/boltzgen/resources/splits/val_monomers_boltzgen_min50_max220.txt'
    result['folding_panel_source_sha256'] = hashlib.sha256(split.read_bytes()).hexdigest()
    (ROOT/'data_manifest.json').write_text(json.dumps(result, indent=2))
    print('DATA_COMPLETE', {k:len(result[k]) for k in ('train','validation','test')}, flush=True)


if __name__ == '__main__': main()
