"""Correct the upstream mean-distance field to the preregistered CA RMSD.

Run only after refolding has completed. Keep all original evaluator CSVs intact;
no inference, training, candidate exclusion, or threshold changes occur here.
"""
import csv
import json
import math
from pathlib import Path

from final_audit import ROOT, OUTPUT, read, recompute_refold_metrics
from tracking import file_hash


def main():
    manifest = read(ROOT / 'benchmark_manifest.json')
    old_scores = read(OUTPUT / 'design_scores.json')
    assert len(old_scores) == 12
    backup = OUTPUT / 'design_scores_vendor_original.json'
    if not backup.exists():
        backup.write_text(json.dumps(old_scores, indent=2))
    records, changes, residuals = [], [], []
    for case in manifest['design_cases']:
        for method in ('native', 'omni'):
            folder = OUTPUT / 'refold' / method / case['case']
            path = folder / 'sequence_scores.csv'
            original = folder / 'sequence_scores_vendor_original.csv'
            if not original.exists():
                original.write_bytes(path.read_bytes())
            with original.open() as handle:
                sequences = list(csv.DictReader(handle))
            assert len(sequences) == 32
            for sequence in sequences:
                candidate = next(r for r in manifest['design'] if r['id'] == Path(sequence['backbone_path']).stem)
                values = recompute_refold_metrics(sequence, candidate, case)
                # Verify the diagnosis against the untouched rounded library output.
                errors = dict(mean_distance=abs(values['mean_ca_distance']-float(sequence['rmsd'])),
                              motif_rmsd=abs(values['motif_rmsd']-float(sequence['motif_rmsd'])))
                assert max(errors.values()) < .0006, (method, candidate['id'], sequence['sample_idx'], errors)
                residuals.append(errors)
                before = sequence['paper_success'] == 'True'
                sequence['vendor_mean_ca_distance'] = sequence['rmsd']
                sequence['vendor_motif_rmsd_rounded'] = sequence['motif_rmsd']
                sequence['rmsd'] = values['rmsd']
                sequence['motif_rmsd'] = values['motif_rmsd']
                confidence = [float(sequence[k]) for k in ('plddt', 'pae')]
                after = (all(math.isfinite(x) for x in confidence) and values['rmsd'] <= 2
                         and values['motif_rmsd'] <= 1 and confidence[0] >= 70 and confidence[1] <= 5)
                sequence['paper_success'] = after
                if before != after:
                    changes.append(dict(method=method, candidate=candidate['id'], sample_idx=sequence['sample_idx'], before=before, after=after))
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sequences[0]))
                writer.writeheader(); writer.writerows(sequences)
            record = next(r for r in old_scores if r['method'] == method and r['case'] == case['case'])
            for scaffold in record['scaffolds']:
                selected = [r for r in sequences if Path(r['backbone_path']).stem == scaffold['id']]
                assert len(selected) == 8
                scaffold.update(success=any(r['paper_success'] for r in selected),
                                best_sc_rmsd=min(r['rmsd'] for r in selected),
                                best_motif_rmsd=min(r['motif_rmsd'] for r in selected))
            record['success'] = sum(r['success'] for r in record['scaffolds'])
            record['metric_correction'] = 'CA RMSD recomputed from PDB; upstream rmsd column was mean distance.'
            (folder / 'scores.json').write_text(json.dumps(record, indent=2))
            records.append(record)
    assert len(residuals) == 384
    (OUTPUT / 'design_scores.json').write_text(json.dumps(records, indent=2))
    audit = dict(sequences=384, all_candidates_retained=True, inference_repeated=False,
                 reason='Upstream data.structure_utils.calc_aligned_rmsd returns mean distance. Final scores use the preregistered root-mean-square CA distance; motif RMSD is also recomputed without CSV rounding.',
                 source_sha256=file_hash(__file__), audit_source_sha256=file_hash(ROOT / 'final_audit.py'),
                 upstream_residuals_max={k:max(r[k] for r in residuals) for k in residuals[0]},
                 changed_sequence_passes=changes, original_scores_sha256=file_hash(backup))
    (OUTPUT / 'design_rescoring.json').write_text(json.dumps(audit, indent=2))
    print('DESIGN_METRIC_CORRECTION', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
