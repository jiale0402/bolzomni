"""Check native exports retain full known chains and valid generated geometry."""
import collections
import hashlib
import math
from pathlib import Path

from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex


def verify_fragment_regrouping(original, exported, generated, missing):
    # Native non-polymer tokenization creates numeric insertion IDs for fragments.
    # A CIF round trip may regroup those fragments without losing any atoms.
    assert all(str(residue[1]).isdigit() for _, residue in missing), missing
    groups = {(chain, residue[0]) for chain, residue in missing}
    assert not any((chain, residue[0]) in groups for chain, residue in generated)

    def signatures(complex_):
        result = collections.Counter()
        for molecule in complex_:
            for block in molecule:
                group = (molecule.id, block.id[0])
                if group not in groups:
                    continue
                for atom in block:
                    row = atom.to_tuple()
                    key = (*group, row[0], row[2], tuple(round(float(x), 3) for x in row[1]))
                    result[key] += 1
        return result

    expected_atoms, actual_atoms = signatures(original), signatures(exported)
    assert expected_atoms and expected_atoms == actual_atoms, 'Known fragment atoms or coordinates changed'
    return {'missing_fragment_ids': missing, 'verified_atoms': sum(expected_atoms.values()),
            'comparison': 'Exact multiset of chain, residue number, atom name, element and coordinates at CIF 0.001 A precision'}


def audit_export(prepared, log, pdb):
    desc = prepared['cplx_desc']
    assert log['id'] == desc.id, 'Export target ID changed'
    positions = desc.pocket_block_ids + desc.lig_block_ids
    generated = {(c, tuple(r)) for (c, r), g in
                 zip(positions, prepared['generate_mask'].tolist()) if g}
    expected = {(m.id, tuple(b.id)) for m in desc.cplx for b in m
                if (m.id, tuple(b.id)) not in generated}
    cif = Path(pdb).with_suffix('.cif')
    assert cif.is_file() and cif.stat().st_size > 0
    exported = mmcif_to_complex(str(cif))
    actual = {(m.id, tuple(b.id)) for m in exported for b in m}
    missing = sorted(expected - actual, key=str)
    regrouping = verify_fragment_regrouping(desc.cplx, exported, generated, missing) if missing else None
    atoms = [a for m in exported for b in m for a in b]
    assert atoms and all(math.isfinite(float(v)) for a in atoms for v in a.coordinate)
    ligand_chains = set(log['lig_chains'])
    assert any(m.id in ligand_chains and len(m) for m in exported), 'Generated chain absent'
    assert Path(pdb).is_file() and Path(pdb).stat().st_size > 0
    return {'complete_known_chains': True, 'known_residues': len(expected),
            'exported_residues': len(actual), 'finite_atoms': len(atoms),
            'target_id_matches': True, 'fragment_regrouping': regrouping,
            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
