"""Independent ESMFold evaluation with fixed thresholds and all-candidate denominators."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ESM_REVISION = '75a3841ee059df2bf4d56688166c8fb459ddd97a'


def align(source, target, mask):
    a, b = source[mask].mean(0), target[mask].mean(0)
    u, _, vt = np.linalg.svd((source[mask]-a).T @ (target[mask]-b))
    rotation = u @ np.diag([1, 1, np.linalg.det(u@vt)]) @ vt
    return (source-a)@rotation+b


def rmsd(x, y, mask):
    return float(np.sqrt(np.mean(np.sum((x[mask]-y[mask])**2, axis=-1))))


def main():
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test', 'validation'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', revision=ESM_REVISION, low_cpu_mem_usage=True).eval().cuda()
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(64)
    tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1', revision=ESM_REVISION)
    refolds = ROOT / 'outputs/refolds'
    refolds.mkdir(exist_ok=True)
    def fold(sequence):
        key = hashlib.sha256(sequence.encode()).hexdigest()
        path = refolds / (key+'.npz')
        if not path.exists():
            ids = tokenizer(sequence, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
            with torch.no_grad():
                pred = model(ids, num_recycles=3)
            ca = pred.positions[-1, 0, :, 1].float().cpu().numpy()
            confidence = pred.plddt[0, :, 1].float().cpu().numpy()
            np.savez_compressed(path, ca=ca, confidence=confidence)
        with np.load(path) as result:
            return result['ca'].copy(), result['confidence'].copy()
    if args.smoke:
        row = json.loads((ROOT/'manifests/data.json').read_text())['validation'][0]
        ca, confidence = fold(row['sequence'])
        assert ca.shape == (len(row['sequence']),3) and np.isfinite(ca).all()
        assert confidence.min() >= 0 and confidence.max() <= 100
        print('ESMFOLD_SMOKE_PASS', row['id'], ca.shape, float(confidence.mean()), flush=True)
        return
    records = []
    for path in sorted((ROOT / 'outputs' / f'generated-{args.split}').glob('*/*.json')):
        r = json.loads(path.read_text())
        result_path = path.with_suffix('.eval.json')
        if path.name.endswith('.eval.json'):
            continue
        if result_path.exists():
            records.append(json.loads(result_path.read_text()))
            continue
        r['pass'] = False
        if r['sequence_valid']:
            folded, confidence = fold(r['sequence'])
            with np.load(path.with_suffix('.npz')) as data:
                ca, native = data['ca'], data['native_ca']
            edit = np.zeros(len(ca), bool)
            edit[r['start']:r['end']+1] = True
            aligned = align(folded, ca, ~edit)
            scaffold_aligned = align(ca, native, ~edit)
            r['local_scrmsd'] = rmsd(aligned, ca, edit)
            r['scaffold_rmsd'] = rmsd(scaffold_aligned, native, ~edit)
            r['native_local_rmsd'] = rmsd(scaffold_aligned, native, edit)
            # HF ESMFold returns probabilities on [0,1]; report pLDDT on [0,100].
            r['local_plddt'] = float(confidence[edit].mean()) * 100.0
            adjacent = np.linalg.norm(np.diff(ca, axis=0), axis=-1)
            # Coarse CA geometry screen, explicitly not an all-atom chemical validation.
            local_bonds = adjacent[max(0,r['start']-1):min(len(adjacent),r['end']+1)]
            distances = np.linalg.norm(ca[:, None]-ca[None, :], axis=-1)
            nonlocal_pairs = np.abs(np.arange(len(ca))[:, None]-np.arange(len(ca))[None, :])>1
            near_edit = edit[:, None] | edit[None, :]
            r['ca_geometry_pass'] = bool(((local_bonds>=3.4)&(local_bonds<=4.2)).all() and not ((distances<2.5)&nonlocal_pairs&near_edit).any())
            r['pass'] = bool(r['local_scrmsd']<2 and r['local_plddt']>=70 and r['scaffold_rmsd']<2 and r['ca_geometry_pass'])
            if r['condition']=='fixed':
                r['pass'] = r['pass'] and r['native_local_rmsd']<2
        result_path.write_text(json.dumps(r, indent=2))
        records.append(r)
        print('EVALUATED', r['id'], r['method'], r['condition'], r['candidate'], r['pass'], flush=True)
    summary = {}
    for condition in ('fixed','flexible'):
        ids = sorted({r['id'] for r in records if r['condition']==condition})
        means = {}
        for method in ('native','feedback'):
            selected = [r for r in records if r['method']==method and r['condition']==condition]
            means[method] = [np.mean([r['pass'] for r in selected if r['id']==i]) for i in ids]
            summary[f'{condition}/{method}'] = {'candidates':len(selected), 'pass_rate':float(np.mean(means[method])),
                'unique_successful_sequences':len({(r['id'],r['sequence']) for r in selected if r['pass']})}
        difference = np.array(means['feedback'])-np.array(means['native'])
        assert np.isfinite(difference).all(), 'Missing paired method outputs'
        rng = np.random.default_rng(42)
        boot = np.array([rng.choice(difference, len(difference), replace=True).mean() for _ in range(5000)])
        summary[condition+'/paired_difference'] = {'mean':float(difference.mean()), 'ci95':np.quantile(boot,[.025,.975]).tolist(),
                                                 'proteins_improved':int((difference>0).sum()), 'proteins_worse':int((difference<0).sum()), 'proteins':len(ids)}
    (ROOT/'outputs'/f'summary-{args.split}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
