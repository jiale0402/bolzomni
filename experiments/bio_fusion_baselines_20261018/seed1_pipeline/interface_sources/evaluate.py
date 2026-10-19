"""Post-hoc CPU-only metrics on immutable completed protein generations."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np
from Bio.PDB import PDBIO, Structure, Model, Chain
import DockQ.DockQ as dockq

ROOT = Path(os.environ['FUSION_ROOT'])
CONFIG = json.loads((ROOT / 'code/config.json').read_text())
sys.path.insert(0, CONFIG['anew_source'])
import native_metrics as native
from residue_mapping import map_sequence_blocks_to_residues
from common import Tracker, atomic_json, sha

METRICS = {
    'DockQ': 'higher', 'iRMSD': 'lower', 'LRMSD': 'lower',
    'fnat': 'higher', 'contact_precision': 'higher', 'F1': 'higher',
    'clashes': 'lower', 'interface_CA_distance_preservation': 'higher',
    'internal_CA_distance_preservation': 'higher',
    'complex_CA_RMSD': 'lower', 'internal_CA_RMSD': 'lower',
    'complex_CA_RMSD_le_2A': 'higher',
}


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_normalized(path, groups, transform=None):
    structure = Structure.Structure('canonical')
    model = Model.Model(0)
    structure.add(model)
    for label, items in groups.items():
        chain = Chain.Chain(label)
        model.add(chain)
        for number, item in sorted(items.items()):
            res = item[1].copy()
            res.detach_parent()
            res.id = (' ', number + 1, ' ')
            for atom in list(res):
                if atom.element in ('H', 'D'):
                    res.detach_child(atom.id)
                elif transform is not None:
                    atom.coord = atom.coord @ transform[0] + transform[1]
            chain.add(res)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))


def distance_preservation(ref_a, gen_a, ref_b, gen_b, internal=False):
    ref = np.linalg.norm(ref_a[:, None] - ref_b[None, :], axis=-1)
    gen = np.linalg.norm(gen_a[:, None] - gen_b[None, :], axis=-1)
    mask = (ref < 15.0)
    if internal:
        mask &= np.triu(np.ones(ref.shape, dtype=bool), 1)
    n = int(mask.sum())
    if n == 0:
        return None, 0
    delta = np.abs(ref[mask] - gen[mask])
    return float(np.mean([np.mean(delta < t) for t in (0.5, 1., 2., 4.)])), n


def clear_dockq_cache():
    for value in vars(dockq).values():
        if callable(getattr(value, 'cache_clear', None)):
            value.cache_clear()


def get_dockq(model_path, reference_path):
    pred = dockq.load_PDB(str(model_path))
    ref = dockq.load_PDB(str(reference_path))
    values, _ = dockq.run_on_all_native_interfaces(
        pred, ref, chain_map={'R': 'R', 'L': 'L'}, no_align=True,
        capri_peptide=False)
    result = values.get('RL')
    clear_dockq_cache()
    return clean(result)


def mapping(row, candidate, pdb):
    mode = 'peptide' if row['benchmark'] == 'lnr' else 'antibody'
    rt, rl = native.select_coordinates(row['reference_pdb'], row, mode)
    gt, gl = native.select_coordinates(pdb, row, mode)
    rt = {item[0]: item for item in rt}
    gt = {item[0]: item for item in gt}
    ref_tokens = native.tokenize_sequence(row['reference_sequence'])
    gen_tokens = native.tokenize_sequence(candidate['gen_seq'])
    assert len(ref_tokens) == len(gen_tokens)
    rl, _ = map_sequence_blocks_to_residues(ref_tokens, rl)
    gl, _ = map_sequence_blocks_to_residues(gen_tokens, gl)
    tc = sorted(rt.keys() & gt.keys())
    lc = sorted(rl.keys() & gl.keys())
    assert len(tc) >= 3 and lc
    # Keep ALL represented native residues, so missing generated contacts do not
    # disappear from the DockQ native-contact denominator. Fixed block numbering
    # prevents sequence identity changes from changing correspondence.
    target_numbers = {key: index for index, key in enumerate(sorted(rt))}
    ref_groups = {'R': {target_numbers[k]: v for k, v in rt.items()}, 'L': rl}
    gen_groups = {'R': {target_numbers[k]: gt[k] for k in tc},
                  'L': {k: gl[k] for k in lc}}
    coords = [np.asarray([v[k][2] for k in keys])
              for v, keys in ((rt, tc), (gt, tc), (rl, lc), (gl, lc))]
    coverage = {'reference_target_CA': len(rt), 'common_target_CA': len(tc),
                'reference_region_CA': len(rl), 'generated_region_CA': len(gl),
                'common_region_CA': len(lc), 'region_blocks': len(ref_tokens),
                'reference_unrepresented_blocks': sorted(set(range(len(ref_tokens))) - rl.keys()),
                'generated_unrepresented_blocks': sorted(set(range(len(gen_tokens))) - gl.keys()),
                'missing_reference_region_CA_in_generation': sorted(rl.keys() - gl.keys())}
    return ref_groups, gen_groups, coords, coverage


def score_candidate(row, candidate, pdb, old, scratch, self_test=False):
    assert sha(pdb) == candidate['pdb_sha256'], 'candidate hash mismatch'
    ref_groups, gen_groups, coords, coverage = mapping(row, candidate, pdb)
    rt, gt, rl, gl = coords
    rot, trans = native.kabsch_transform(gt, rt)
    lrot, ltrans = native.kabsch_transform(gl, rl)
    pose = native.rmsd(gl @ rot + trans, rl)
    internal = native.rmsd(gl @ lrot + ltrans, rl)
    residual = abs(pose - old['scores']['structural']['complex_RMSD'])
    if row['benchmark'] == 'lnr':
        residual = max(residual, abs(internal - old['scores']['structural']['peptide_RMSD']))
    assert residual < 1e-6, f'original CA metric reproduction failed: {residual}'
    ref_path, gen_path = Path(scratch) / 'ref.pdb', Path(scratch) / 'gen.pdb'
    write_normalized(ref_path, ref_groups)
    write_normalized(gen_path, gen_groups)
    info = get_dockq(gen_path, ref_path)
    metrics = {key: None for key in METRICS}
    errors = {}
    if info is None:
        errors['DockQ'] = 'No native heavy-atom contact below 5 A in the designated interface'
    else:
        for key in ('DockQ', 'iRMSD', 'LRMSD', 'fnat', 'F1', 'clashes'):
            metrics[key] = info[key]
        metrics['contact_precision'] = (info['nat_correct'] / info['model_total']
                                        if info['model_total'] else None)
        if not info['model_total']:
            errors['contact_precision'] = 'No generated heavy-atom contact below 5 A'
    interface_dist, interface_n = distance_preservation(rl, gl, rt, gt)
    internal_dist, internal_n = distance_preservation(rl, gl, rl, gl, internal=True)
    metrics.update(interface_CA_distance_preservation=interface_dist,
                   internal_CA_distance_preservation=internal_dist,
                   complex_CA_RMSD=pose, internal_CA_RMSD=internal,
                   complex_CA_RMSD_le_2A=float(pose <= 2.0))
    record = {'metrics': metrics, 'dockq_details': info, 'coverage': coverage,
              'distance_pair_denominators': {'interface': interface_n, 'internal': internal_n},
              'errors': errors, 'original_CA_reproduction_max_abs_error': residual,
              'candidate_pdb_sha256': candidate['pdb_sha256']}
    if self_test:
        identity = get_dockq(ref_path, ref_path)
        assert identity and abs(identity['DockQ'] - 1.) < 1e-10
        assert abs(identity['fnat'] - 1.) < 1e-10
        angle = .72
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        write_normalized(Path(scratch) / 'rot.pdb', ref_groups,
                         (rotation, np.array([13., -7., 5.])))
        rigid = get_dockq(Path(scratch) / 'rot.pdb', ref_path)
        assert abs(rigid['DockQ'] - 1.) < 2e-3
        shifted = dict(ref_groups)
        shifted['L'] = {}
        for key, item in ref_groups['L'].items():
            copied = item[1].copy()
            for atom in copied.get_atoms():
                atom.coord += np.array([50., 50., 50.])
            shifted['L'][key] = (item[0], copied, item[2])
        write_normalized(Path(scratch) / 'shift.pdb', shifted)
        damaged = get_dockq(Path(scratch) / 'shift.pdb', ref_path)
        assert damaged['fnat'] == 0 and damaged['DockQ'] < .1
        record['self_test'] = {'identity': identity, 'rigid_motion': rigid,
                               'separated_design_region': damaged}
    return record


def score_target(task):
    model, row = task
    started = time.monotonic()
    root = Path(CONFIG['models'][model]['benchmark_root'])
    directory = root / 'scores' / row['benchmark'] / row['id']
    candidates = jsonl(directory / 'candidate_input/results.jsonl')
    old = {r['n']: r for r in jsonl(directory / 'evaluation.jsonl')}
    assert len(candidates) == 30 and {r['n'] for r in candidates} == set(range(30))
    out = ROOT / 'targets' / model / row['benchmark'] / (row['id'] + '.json')
    if out.exists():
        result = json.loads(out.read_text())
        assert result['protocol_sha256'] == sha(ROOT / 'code/protocol.json')
        assert len(result['records']) == 30
        for c in candidates:
            pdb = directory / 'candidate_input' / row['id'] / (str(c['n']) + '.pdb')
            assert sha(pdb) == c['pdb_sha256']
        return result
    records = []
    with tempfile.TemporaryDirectory(prefix='extended-protein-', dir=os.getenv('TMPDIR')) as scratch:
        for candidate in sorted(candidates, key=lambda x: x['n']):
            n = candidate['n']
            base = {'model': model, 'benchmark': row['benchmark'], 'target': row['id'], 'n': n}
            pdb = directory / 'candidate_input' / row['id'] / (str(n) + '.pdb')
            try:
                assert candidate['success'] is True and old[n]['success'] is True
                base.update(score_candidate(row, candidate, pdb, old[n], scratch,
                                            self_test=(row['index'] == 0 and n == 0)))
            except Exception:
                base.update(metrics={key: None for key in METRICS},
                            errors={'scoring_exception': traceback.format_exc()})
            records.append(base)
    result = {'model': model, 'benchmark': row['benchmark'], 'target': row['id'],
              'protocol_sha256': sha(ROOT / 'code/protocol.json'),
              'reference_pdb_sha256': sha(row['reference_pdb']),
              'records': records, 'elapsed_seconds': time.monotonic() - started}
    atomic_json(out, result)
    return result


def main():
    started = time.monotonic()
    import importlib.metadata
    assert importlib.metadata.version('DockQ') == '2.1.3'
    assert all(sha(ROOT / 'code' / k) == v for k, v in json.loads((ROOT / 'DEPLOYED.json').read_text()).items())
    tracker = Tracker('extra-protein-metrics', {'protocol_sha256': sha(ROOT / 'code/protocol.json')})
    try:
        tasks = []
        evidence = {'models': {}, 'manifests': {}, 'software': {'DockQ': '2.1.3', 'numpy': np.__version__}}
        for model, config in CONFIG['models'].items():
            path = Path(config['benchmark_root']) / 'run.json'
            run = json.loads(path.read_text())
            assert run['run_hash'] == config['run_hash']
            assert run['checkpoint_sha256'] == config['checkpoint_sha256']
            assert sha(run['checkpoint']) == config['checkpoint_sha256']
            evidence['models'][model] = {'run_json_sha256': sha(path), **config}
            for spec in run['benchmark_specs']:
                if spec['name'] not in ('lnr', 'rabd'):
                    continue
                path = Path(spec['manifest'])
                assert sha(path) == spec['manifest_sha256']
                rows = jsonl(path)
                assert len(rows) == {'lnr': 93, 'rabd': 60}[spec['name']]
                evidence['manifests'][model + '/' + spec['name']] = {'sha256': sha(path), 'records': len(rows)}
                tasks.extend((model, row) for row in rows)
        atomic_json(ROOT / 'INPUT_AUDIT.json', evidence)
        results = []
        workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '16'))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_target, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                counts = {metric: [r['metrics'][metric] for r in result['records']
                                   if r['metrics'].get(metric) is not None] for metric in METRICS}
                prefix = result['model'] + '/' + result['benchmark']
                record = {'targets_done': len(results), 'target': result['target'], 'model': result['model'],
                          'benchmark': result['benchmark'], 'candidates': 30,
                          'scoring_exceptions': sum('scoring_exception' in r['errors'] for r in result['records'])}
                record.update({prefix + '/' + k: float(np.mean(v)) for k, v in counts.items() if v})
                tracker.log(record)
                atomic_json(ROOT / 'PROGRESS.json', {'targets_done': len(results), 'targets_expected': 306,
                                                     'latest': record, 'updated_unix': time.time()})
                print(json.dumps(record), flush=True)
        records = [r for t in sorted(results, key=lambda t: (t['model'], t['benchmark'], t['target'])) for r in t['records']]
        assert len(records) == 9180
        (ROOT / 'candidates.jsonl').write_text(''.join(json.dumps(clean(r), allow_nan=False) + '\n' for r in records))
        exceptions = [r for r in records if 'scoring_exception' in r['errors']]
        summary = {'targets': len(results), 'candidates': len(records), 'scoring_exceptions': len(exceptions),
                   'elapsed_seconds': time.monotonic() - started, 'allocated_cpus': workers,
                   'candidate_jsonl_sha256': sha(ROOT / 'candidates.jsonl'),
                   'protocol_sha256': sha(ROOT / 'code/protocol.json'), 'checkpoint_selection_changed': False,
                   'training_resumed': False, 'new_generation_attempts': 0}
        atomic_json(ROOT / 'SCORING_COMPLETE.json', summary)
        tracker.finish(summary, [ROOT / 'SCORING_COMPLETE.json', ROOT / 'INPUT_AUDIT.json',
                                  ROOT / 'candidates.jsonl', ROOT / 'code/protocol.json'])
    except Exception:
        atomic_json(ROOT / 'FAILED.json', {'traceback': traceback.format_exc(), 'time': time.time()})
        tracker.run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
