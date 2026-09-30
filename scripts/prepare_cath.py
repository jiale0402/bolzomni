"""Prepare a deterministic small CATH topology-split panel and original structures."""
import copy
import argparse
import hashlib
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Structure, Model, Chain
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/'


def download(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=25) as response:
            path.with_suffix(path.suffix + '.tmp').write_bytes(response.read())
        path.with_suffix(path.suffix + '.tmp').replace(path)


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--train-count', type=int, default=200)
    ap.add_argument('--validation-count', type=int, default=20)
    ap.add_argument('--test-count', type=int, default=20)
    ap.add_argument('--max-length', type=int, default=160)
    ap.add_argument('--exclude-test-manifest', type=Path)
    ap.add_argument('--reuse-verified-log', type=Path)
    args = ap.parse_args()
    ROOT = args.root.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'manifests').mkdir(exist_ok=True)
    counts = [('test',args.test_count),('validation',args.validation_count),('train',args.train_count)]
    excluded_test = set()
    if args.exclude_test_manifest:
        excluded_test = {r['id'] for r in json.loads(args.exclude_test_manifest.read_text())['test']}
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    (data / 'raw').mkdir(exist_ok=True)
    download(BASE + 'splits.json', data / 'splits.json')
    splits = json.loads((data / 'splits.json').read_text())
    membership = {name: split for split, names in splits.items() for name in names}
    assert sum(map(len, splits.values())) == len(membership), 'Overlapping source splits'
    pools = {s: [] for s in ('train', 'validation', 'test')}
    print('READ_CATH', flush=True)
    with urllib.request.urlopen(BASE + 'chain_set.jsonl', timeout=120) as response:
        for line in response:
            row = json.loads(line)
            split = membership.get(row['name'])
            if split == 'val':
                split = 'validation'
            if split not in pools or not 80 <= len(row['seq']) <= args.max_length:
                continue
            if split == 'test' and row['name'] in excluded_test:
                continue
            if set(row['seq']) - set('ACDEFGHIKLMNPQRSTVWY'):
                continue
            coords = np.asarray([row['coords'][a] for a in ('N', 'CA', 'C', 'O')])
            if not np.isfinite(coords).all():
                continue
            # Uniform hash priority, independent of model performance.
            row['priority'] = hashlib.sha256(('omni-poc-20260926-' + row['name']).encode()).hexdigest()
            pools[split].append(row)
    print('ELIGIBLE', {k: len(v) for k, v in pools.items()}, flush=True)
    candidates = []
    for split, count in counts:
        candidates.extend(r['name'][:4].lower() for r in sorted(pools[split], key=lambda r: r['priority'])[:count * 3])
    def prefetch(pdb_id):
        try:
            download(f'https://files.rcsb.org/download/{pdb_id.upper()}.cif', data / 'raw' / (pdb_id + '.cif'))
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(prefetch, sorted(set(candidates))))
    print('PREFETCH_COMPLETE', flush=True)
    manifest = {s: [] for s in pools}
    failures, used_sequences, used_pdbs = [], set(), set()
    parser = MMCIFParser(QUIET=True)
    verified = set()
    previously_excluded = {}
    if args.reuse_verified_log:
        for line in args.reuse_verified_log.read_text().splitlines():
            fields = line.split()
            if len(fields) == 4 and fields[0] == 'PREPARED':
                verified.add((fields[1], fields[3]))
            elif len(fields) > 2 and fields[0] == 'EXCLUDED':
                reason = ' '.join(fields[2:])
                if reason in ('Backbone break or non-trans geometry outside initial scope',
                              'No exact contiguous canonical sequence match in first model',
                              'Original/source coordinate mapping mismatch'):
                    previously_excluded[fields[1]] = reason
    # Reserve held-out identities first; no cross-split exact sequence/PDB overlap.
    for split, count in counts:
        for row in sorted(pools[split], key=lambda r: r['priority']):
            if len(manifest[split]) == count:
                break
            name, sequence = row['name'], row['seq']
            if name in previously_excluded:
                failures.append({'id': name, 'split': split, 'error': previously_excluded[name],
                                 'reused_exclusion': True})
                continue
            pdb_id = name[:4].lower()
            if sequence in used_sequences or pdb_id in used_pdbs:
                continue
            try:
                prepared = data / split / name / 'native.pdb'
                reused = (split, name) in verified and prepared.exists()
                if reused:
                    # Recheck sequence, source coordinates and geometry below. Avoid
                    # reparsing the large source complex for an already verified chain.
                    structure = PDBParser(QUIET=True).get_structure(name, prepared)
                else:
                    raw_path = data / 'raw' / (pdb_id + '.cif')
                    download(f'https://files.rcsb.org/download/{pdb_id.upper()}.cif', raw_path)
                    structure = parser.get_structure(pdb_id, raw_path)
                matches = []
                for chain in structure[0]:
                    residues = [r for r in chain if r.id[0] == ' ']
                    observed = ''.join(seq1(r.resname, undef_code='X') for r in residues)
                    start = observed.find(sequence)
                    if start >= 0:
                        selected = residues[start:start + len(sequence)]
                        if all(all(a in r for a in ('N', 'CA', 'C', 'O')) for r in selected):
                            matches.append((chain.id, selected))
                if not matches:
                    raise ValueError('No exact contiguous canonical sequence match in first model')
                # Check geometry, not just the sequence, against the source record.
                ref = np.asarray(row['coords']['CA'])
                chain_id, residues = min(matches, key=lambda pair: np.mean((np.asarray([r['CA'].coord for r in pair[1]]) - ref) ** 2))
                ca = np.asarray([r['CA'].coord for r in residues])
                if np.sqrt(np.mean(np.sum((ca - ref) ** 2, axis=-1))) > 0.2:
                    raise ValueError('Original/source coordinate mapping mismatch')
                if np.any((np.linalg.norm(np.diff(ca, axis=0), axis=-1) < 3.4) | (np.linalg.norm(np.diff(ca, axis=0), axis=-1) > 4.2)):
                    raise ValueError('Backbone break or non-trans geometry outside initial scope')
                dest = data / split / name
                dest.mkdir(parents=True, exist_ok=True)
                clean = Structure.Structure(name)
                model = Model.Model(0)
                out_chain = Chain.Chain('A')
                clean.add(model)
                model.add(out_chain)
                for i, residue in enumerate(residues, 1):
                    r = residue.copy()
                    r.detach_parent()
                    r.id = (' ', i, ' ')
                    out_chain.add(r)
                writer = PDBIO()
                writer.set_structure(clean)
                writer.save(str(dest / 'native.pdb'))
                rng = random.Random(row['priority'])
                length = rng.choice([10, 12, 16])
                start = rng.randint(6, len(sequence) - length - 6)
                end = start + length - 1
                for condition in ('fixed', 'flexible'):
                    entry = {'path': str(dest / 'native.pdb'), 'include': [{'chain': {'id': 'A'}}],
                             'design': [{'chain': {'id': 'A', 'res_index': f'{start + 1}..{end + 1}'}}]}
                    if condition == 'flexible':
                        entry['structure_groups'] = [
                            {'group': {'id': 'A', 'visibility': 1, 'res_index': f'1..{start}'}},
                            {'group': {'id': 'A', 'visibility': 1, 'res_index': f'{end + 2}..{len(sequence)}'}}]
                    (dest / (condition + '.yaml')).write_text(yaml.safe_dump({'entities': [{'file': entry}]}, sort_keys=False))
                record = {'id': name, 'pdb': pdb_id, 'source_chain': chain_id, 'sequence': sequence,
                          'start': start, 'end': end, 'length': len(sequence), 'directory': str(dest)}
                if reused:
                    record['source_chain'] = None
                    record['revalidated_prepared_chain'] = 'A'
                    record['preparation_log_sha256'] = hashlib.sha256(args.reuse_verified_log.read_bytes()).hexdigest()
                manifest[split].append(record)
                used_sequences.add(sequence)
                used_pdbs.add(pdb_id)
                print('PREPARED', split, len(manifest[split]), name, flush=True)
            except Exception as error:
                failures.append({'id': name, 'split': split, 'error': str(error)})
                print('EXCLUDED', name, str(error), flush=True)
        if len(manifest[split]) != count:
            raise RuntimeError(f'Insufficient usable {split} structures: {len(manifest[split])}/{count}')
    (ROOT / 'manifests' / 'data.json').write_text(json.dumps(manifest, indent=2))
    (ROOT / 'manifests' / 'data_exclusions.json').write_text(json.dumps(failures, indent=2))
    print('DATA_COMPLETE', {k: len(v) for k, v in manifest.items()}, flush=True)


if __name__ == '__main__':
    main()
