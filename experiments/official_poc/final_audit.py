"""Independent completion, denominator, provenance, W&B and Slurm accounting audit."""
import csv
import datetime
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import wandb
from tracking import Tracker, file_hash

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OUTPUT = ROOT / 'results'
RUN = ROOT / 'runs/corrected-1729'
ENTITY_PROJECT = 'jialechen2025-northwestern-university/boltz-omni-joint'


def read(path):
    return json.loads(path.read_text())


def independent_ca_lddt(backbone, row):
    """Direct pair-distance definition; independent of Biotite's implementation."""
    if not np.isfinite(backbone).all():
        return 0.0
    indices = sorted(map(int, row['resolved_ca']))
    # Biotite AtomArray stores float32 coordinates and includes bin boundaries.
    # Match those public metric semantics without calling its scoring function.
    true = np.asarray([row['resolved_ca'][str(i)] for i in indices], dtype=np.float32)
    pred = backbone[np.asarray(indices)-1, 1].astype(np.float32)
    true_dist = np.linalg.norm(true[:, None] - true[None, :], axis=-1)
    pred_dist = np.linalg.norm(pred[:, None] - pred[None, :], axis=-1)
    mask = (true_dist <= 15) & ~np.eye(len(true), dtype=bool)
    assert mask.any()
    delta = np.abs(pred_dist - true_dist)[mask]
    return float(np.mean([np.mean(delta <= threshold) for threshold in (.5, 1, 2, 4)]))


def check_design_inputs_and_files(manifest):
    checked = 0
    for row in manifest['design']:
        assert file_hash(row['yaml']) == row['input_sha256']
        assert file_hash(Path(row['yaml']).with_name('prompt.pdb')) == row['prompt_sha256']
        case = next(c for c in manifest['design_cases'] if c['case'] == row['case'])
        reference = Path(case['reference'])
        assert file_hash(reference) == case['reference_sha256']
        segments = {}
        for line in reference.read_text().splitlines():
            if line.startswith('ATOM'):
                segments.setdefault(line[21], {})[int(line[22:26])] = line[17:20]
        cursor, expected = 1, {}
        for part in row['motif_placements'].split('/'):
            if part.isdigit():
                cursor += int(part)
            else:
                for _, aa in sorted(segments[part].items()):
                    expected[cursor] = aa
                    cursor += 1
        assert set(expected) == set(row['motif_indices']) and cursor-1 == row['length']
        for method in ('native', 'omni'):
            path = OUTPUT / 'scaffolds' / method / row['case'] / (row['id'] + '.pdb')
            residues, atoms = {}, {}
            for line in path.read_text().splitlines():
                if line.startswith('ATOM'):
                    i = int(line[22:26])
                    residues[i] = line[17:20]
                    atoms.setdefault(i, set()).add(line[12:16].strip())
            assert set(residues) == set(range(1, row['length']+1))
            assert all(residues[i] == aa for i, aa in expected.items())
            assert all(names == {'N', 'CA', 'C', 'O'} for names in atoms.values())
            checked += 1
    assert checked == 48
    return checked


def pdb_coordinates(path):
    coordinates = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('ATOM') and line[16] in (' ', 'A'):
            key = (line[21], int(line[22:26]), line[12:16].strip())
            coordinates[key] = [float(line[a:a+8]) for a in (30, 38, 46)]
    return coordinates


def independent_aligned_distances(x, y):
    """Rigid Kabsch distances directly from PDB coordinates, without evaluator imports."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    x = x - x.mean(axis=0); y = y - y.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3); correction[-1, -1] = np.linalg.det(u @ vt)
    return np.linalg.norm(x @ u @ correction @ vt - y, axis=1)


def independent_rmsd(x, y):
    return float(np.sqrt(np.mean(independent_aligned_distances(x, y)**2)))


def recompute_refold_metrics(sequence, candidate, case):
    scaffold = pdb_coordinates(sequence['backbone_path'])
    folded = pdb_coordinates(sequence['sample_path'])
    reference = pdb_coordinates(case['reference'])
    ca_keys = [('A', i, 'CA') for i in range(1, candidate['length']+1)]
    distances = independent_aligned_distances([scaffold[k] for k in ca_keys], [folded[k] for k in ca_keys])
    rmsd = float(np.sqrt(np.mean(distances**2)))
    cursor, true_motif, pred_motif = 1, [], []
    for segment in candidate['motif_placements'].split('/'):
        if segment.isdigit():
            cursor += int(segment)
        else:
            residues = sorted({i for chain, i, atom in reference if chain == segment})
            for i in residues:
                for atom in ('N', 'CA', 'C', 'O'):
                    true_motif.append(reference[(segment, i, atom)])
                    pred_motif.append(folded[('A', cursor, atom)])
                cursor += 1
    motif_rmsd = independent_rmsd(true_motif, pred_motif)
    return dict(rmsd=rmsd, motif_rmsd=motif_rmsd, mean_ca_distance=float(distances.mean()))


def check_refold_rmsds(sequence, candidate, case):
    values = recompute_refold_metrics(sequence, candidate, case)
    errors = {key: abs(values[key]-float(sequence[key])) for key in ('rmsd', 'motif_rmsd')}
    # The upstream evaluator saves metrics rounded to three decimal places.
    assert max(errors.values()) < .0006, (candidate['id'], sequence['sample_idx'], errors)
    return errors


def main():
    # The upstream evaluator's "rmsd" is mean distance, not root-mean-square.
    # Recompute the locked metric from existing PDBs only after GPU evaluation ends.
    # Preserve every upstream CSV and record this correction before final reporting.
    import rescore_design
    import report
    rescore_design.main()
    report.main()
    data = read(ROOT / 'data_manifest.json')
    manifest = read(ROOT / 'benchmark_manifest.json')
    cfg = read(RUN / 'config.json')
    assert [len(data[k]) for k in ('train', 'validation', 'test')] == [128, 16, 8]
    all_rows = [r for k in ('train', 'validation', 'test') for r in data[k]]
    assert len({r['id'] for r in all_rows}) == 152
    assert len({r['sequence'] for r in all_rows}) == 152
    assert len({r['official_record']['chains'][0]['cluster_id'] for r in all_rows}) == 152
    for row in all_rows:
        assert file_hash(ROOT / 'data' / row['archive_member']) == row['member_sha256']
        assert file_hash(Path(row['directory']) / 'native.pdb') == row['pdb_sha256']
    assert file_hash(ROOT / 'data_manifest.json') == cfg['data_manifest_sha256']
    original_checkpoint = Path('/projects/p53127/anewomni_work/boltz_adapter_20260922/assets/boltzgen1_diverse.ckpt')
    assert file_hash(original_checkpoint) == cfg['base_checkpoint_sha256']
    training_sources = [ROOT / 'train.py', ROOT / 'corrected_model.py']
    training_sources += [PROJECT / 'experiments/omni_joint' / name for name in
                         ('model.py', 'full_model.py', 'cache_training.py')]
    for path in training_sources:
        assert file_hash(path) == cfg['sources'][str(path.relative_to(PROJECT))], path
    completion = read(RUN / 'completion.json')
    assert completion['training_completed'] and completion['steps'] == 400
    journal = [json.loads(line) for line in (RUN / 'metrics.jsonl').read_text().splitlines()]
    updates = [r for r in journal if 'train/loss' in r]
    assert [r['optimizer_step'] for r in updates] == list(range(1, 401))
    assert all(math.isfinite(v) for r in updates for k, v in r.items() if isinstance(v, (int, float)))
    checkpoint_hash = file_hash(RUN / 'final.pt')
    summary = read(OUTPUT / 'summary.json')
    assert summary['checkpoint_sha256'] == checkpoint_hash
    samples = []
    for path in OUTPUT.glob('*.json'):
        if path.name.endswith(('-native.json', '-omni.json')):
            samples.append(read(path))
    assert len(samples) == 128
    design_files_verified = check_design_inputs_and_files(manifest)
    ca_lddt_errors = []
    for row in manifest['folding']:
        pairs = {}
        for method in ('native', 'omni'):
            chosen = [r for r in samples if r['mode'] == 'folding' and r['target'] == row['id'] and r['method'] == method]
            assert len(chosen) == 5 and {r['candidate'] for r in chosen} == set(range(5))
            pairs[method] = {(r['candidate'], r['seed']) for r in chosen}
            assert all(math.isfinite(r[k]) and 0 <= r[k] <= 1.00001 for r in chosen for k in ('ca_lddt', 'tm_score', 'all_atom_lddt'))
            for record in chosen:
                backbone = np.load(OUTPUT / f'{record["id"]}-{method}.npz')['backbone']
                assert backbone.shape == (row['length'], 4, 3)
                assert bool(np.isfinite(backbone).all()) == record['finite_geometry']
                error = abs(independent_ca_lddt(backbone, row) - record['ca_lddt'])
                assert error < 1e-6, (record['id'], method, error)
                ca_lddt_errors.append(error)
        assert pairs['native'] == pairs['omni']
    sequence_count = 0
    design_rmsd_errors = []
    for case in manifest['design_cases']:
        pairs = {}
        for method in ('native', 'omni'):
            chosen = [r for r in samples if r['mode'] == 'design' and r['task'] == case['case'] and r['method'] == method]
            assert len(chosen) == 4 and {r['candidate'] for r in chosen} == set(range(4))
            pairs[method] = {(r['candidate'], r['seed']) for r in chosen}
            folder = OUTPUT / 'refold' / method / case['case']
            with (folder / 'sequence_scores.csv').open() as handle:
                sequences = list(csv.DictReader(handle))
            assert len(sequences) == 32
            scores = read(folder / 'scores.json')
            for candidate in chosen:
                selected = [r for r in sequences if Path(r['backbone_path']).stem == candidate['id']]
                assert len(selected) == 8 and len({r['sample_idx'] for r in selected}) == 8
                passed = []
                for row in selected:
                    values = {k: float(row[k]) for k in ('rmsd', 'motif_rmsd', 'plddt', 'pae')}
                    success = all(math.isfinite(v) for v in values.values()) and values['rmsd'] <= 2 and values['motif_rmsd'] <= 1 and values['plddt'] >= 70 and values['pae'] <= 5
                    assert success == (row['paper_success'] == 'True')
                    assert Path(row['sample_path']).is_file()
                    prompt = next(r for r in manifest['design'] if r['id'] == candidate['id'])
                    design_rmsd_errors.append(check_refold_rmsds(row, prompt, case))
                    passed.append(success)
                scaffold = next(r for r in scores['scaffolds'] if r['id'] == candidate['id'])
                assert scaffold['success'] == any(passed)
            sequence_count += len(sequences)
        assert pairs['native'] == pairs['omni']
    assert sequence_count == 384
    assert all(r['checkpoint_sha256'] == checkpoint_hash for r in samples if r['method'] == 'omni')

    submissions = [read(p) for p in (ROOT / 'submissions').glob('*.json')]
    job_ids = sorted(r['job_id'].split(';')[0] for r in submissions)
    fields = ['JobIDRaw', 'JobName', 'State', 'ExitCode', 'ElapsedRaw', 'AllocTRES', 'Account', 'NodeList', 'Start', 'End']
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(job_ids), '--format=' + ','.join(fields)], text=True)
    (OUTPUT / 'slurm_accounting.txt').write_text(raw)
    jobs = [dict(zip(fields, line.split('|'))) for line in raw.splitlines() if line.strip()]
    gpu_jobs = []
    for job in jobs:
        resources = dict(v.split('=', 1) for v in job['AllocTRES'].split(',') if '=' in v)
        count = int(resources.get('gres/gpu', 0))
        if count:
            assert job['State'] not in ('RUNNING', 'PENDING', 'COMPLETING'), job
            job['gpu_hours'] = int(job['ElapsedRaw']) * count / 3600
            gpu_jobs.append(job)
    assert {'7664164', '7664745', '7665712', '7666088', '7665272', '7665276'} <= {j['JobIDRaw'] for j in gpu_jobs}
    for record in submissions:
        eligible = [a for a, r in record['eligibility'].items() if r['eligible']]
        assert record['account'] == max(eligible, key=lambda a: record['user_fairshare'][a])

    api = wandb.Api(timeout=45)
    remote_runs = {}
    for label, folder in [('training', RUN), ('generation', OUTPUT / 'generation-tracking'),
                          ('refold', OUTPUT / 'refold-tracking'), ('report', OUTPUT / 'report-tracking')]:
        status = read(folder / 'tracking_status.json')
        assert status['mode'] == 'online' and not status['sync_required'], (label, status)
        run = api.run(ENTITY_PROJECT + '/' + status['id'])
        assert run.state == 'finished', (label, run.state)
        remote_runs[label] = dict(id=run.id, state=run.state, url=run.url, optimizer_step=run.summary.get('optimizer_step'))
        if label in ('generation', 'refold'):
            source = ROOT / ('generate.py' if label == 'generation' else 'refold.py')
            assert run.config['source_sha256'] == file_hash(source)
            assert run.config['manifest_sha256'] == file_hash(ROOT / 'benchmark_manifest.json')
            remote_runs[label]['source_sha256_verified'] = file_hash(source)
        if label == 'generation':
            assert run.config['checkpoint_sha256'] == checkpoint_hash
        if label == 'training':
            assert run.summary['optimizer_step'] == 400
            models = [a for a in run.logged_artifacts() if a.type == 'model']
            assert any(a.metadata.get('sha256') == checkpoint_hash for a in models)
            remote_runs[label]['checkpoint_artifact_sha256_verified'] = checkpoint_hash
    audit = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        complete=True, counts=dict(train=128, validation=16, folding_predictions=80, scaffolds=48, refolded_sequences=384),
        verified_training_sources={str(p.relative_to(PROJECT)): file_hash(p) for p in training_sources},
        data_hashes_verified=True, original_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        updates=400, paired_seeds_verified=True,
        all_thresholds_recomputed=True, design_input_mappings_and_scaffold_files_verified=design_files_verified,
        independent_ca_lddt_predictions=len(ca_lddt_errors),
        independent_ca_lddt_max_error=max(ca_lddt_errors),
        independent_refold_rmsd_sequences=len(design_rmsd_errors),
        independent_refold_rmsd_max_errors={key: max(r[key] for r in design_rmsd_errors)
                                            for key in ('rmsd', 'motif_rmsd')},
        design_metric_correction=read(OUTPUT / 'design_rescoring.json'),
        checkpoint_sha256=checkpoint_hash, wandb=remote_runs,
        allocated_GPU_hours=sum(j['gpu_hours'] for j in gpu_jobs), gpu_jobs=gpu_jobs,
        accounting_scope='All official_poc submitted GPU jobs, including three zero-update preflight failures; allocated wall time, not utilization.',
        fairshare_selection_verified=True)
    (OUTPUT / 'final_audit.json').write_text(json.dumps(audit, indent=2))
    tracker = Tracker(OUTPUT / 'report-tracking', {'completion_audit': True}, 'official-data-poc-final-report', job_type='evaluation')
    try:
        tracker.log({'audit/complete': 1, 'compute/allocated_GPU_hours_including_failed_preflights': audit['allocated_GPU_hours']}, 400)
        artifact = wandb.Artifact('official-data-poc-completion-audit', type='diagnostics')
        artifact.add_file(str(OUTPUT / 'final_audit.json'))
        artifact.add_file(str(OUTPUT / 'slurm_accounting.txt'))
        artifact.add_file(str(OUTPUT / 'design_rescoring.json'))
        for path in sorted((OUTPUT / 'refold').glob('*/*/sequence_scores.csv')):
            artifact.add_file(str(path), name=str(path.relative_to(OUTPUT)))
        for name in ('training_diagnostics.json', 'training_diagnostics.png'):
            if (OUTPUT / name).exists():
                artifact.add_file(str(OUTPUT / name))
        tracker.run.log_artifact(artifact)
    finally:
        tracker.finish(True)
    print('COMPLETION_AUDIT', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
