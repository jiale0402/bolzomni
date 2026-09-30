"""Compile and cache the static prompt through native BoltzGen trunk once."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals
from boltzgen.task.predict.data_from_yaml import Dataset, PredictionDataset, collate
from poc_model import ROOT, ASSETS, load_model, move, safe_prompt, labels


class Captured(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    rows = json.loads((ROOT / 'manifests/data.json').read_text())
    model = load_model()
    moldir = str(ASSETS / 'mols.zip')
    canonicals = load_canonicals(moldir)
    (ROOT / 'cache/features').mkdir(parents=True, exist_ok=True)
    captured = {}
    def capture(**kwargs):
        captured.update({k: v for k, v in kwargs.items() if k in ('s_trunk', 's_inputs', 'feats', 'diffusion_conditioning')})
        raise Captured()
    model.structure_module.sample = capture
    total = 0
    for split, records in rows.items():
        for row in records[:args.limit or None]:
            for condition in ('fixed', 'flexible'):
                dest = ROOT / 'cache/features' / f'{row["id"]}-{condition}.pt'
                if dest.exists():
                    continue
                seed = int.from_bytes(row['id'].encode(), 'little') % (2**31)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                dataset = PredictionDataset(
                    Dataset(str(Path(row['directory']) / (condition + '.yaml')), Tokenizer(atomize_modified_residues=False), Featurizer()),
                    canonicals, moldir, atom14=True, design=True, disulfide_on=False,
                )
                default_rng = np.random.default_rng
                np.random.default_rng = lambda value=None: default_rng(seed if value is None else value)
                try:
                    f = move(collate([dataset[0]]), 'cuda')
                finally:
                    np.random.default_rng = default_rng
                assert int(f['design_mask'].sum()) == row['end'] - row['start'] + 1
                assert f['res_type'].shape[1] == row['length']
                target = f['coords'].reshape(1, -1, 3).clone()
                target_labels = labels(f)
                prompt = safe_prompt(model, f, condition)
                # Counterfactual changes to hidden labels, side-chain features, and
                # target-only arrays must not change the compiled static prompt.
                other = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in f.items()}
                dm = other['design_mask'].bool()
                am = torch.bmm(other['atom_to_token'].float(), dm.float()[..., None]).squeeze(-1).bool()
                side = am & ~other['backbone_mask'].bool()
                for k in ('res_type', 'profile', 'ccd', 'res_type_clone'):
                    other[k][dm] = 0
                other['msa'].masked_fill_(dm[:, None, :], 0)
                for k in ('ref_element', 'ref_charge', 'ref_atom_name_chars'):
                    other[k][side] = 0
                other['ref_pos'][am] = 777
                other['ref_chirality'][am] = 0
                for k in ('coords', 'bfactor', 'plddt', 'fake_atom_mask', 'atom_resolved_mask'):
                    other[k] = torch.zeros_like(other[k])
                if condition == 'flexible':
                    other['center_coords'][dm] = 777
                counterfactual = safe_prompt(model, other, condition)
                changed = [k for k, v in prompt.items() if isinstance(v, torch.Tensor) and not torch.equal(v, counterfactual[k])]
                assert not changed, ('Hidden input leakage', changed)
                captured.clear()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    try:
                        model(prompt, recycling_steps=3)
                    except Captured:
                        pass
                assert captured, 'Trunk capture failed'
                # Diffusion runs in float32, as in native forward.
                captured['s_trunk'] = captured['s_trunk'].float()
                captured['s_inputs'] = captured['s_inputs'].float()
                result = {'conditioning': move(captured, 'cpu'), 'target': target.cpu(),
                          'labels': target_labels.cpu(), 'resolved': f['atom_resolved_mask'].cpu(),
                          'condition': condition, 'row': row, 'split': split, 'hidden_feature_audit': True}
                torch.save(result, str(dest) + '.tmp')
                Path(str(dest) + '.tmp').replace(dest)
                total += 1
                print('CACHE', total, split, row['id'], condition, flush=True)
    print('CACHE_COMPLETE', total, flush=True)


if __name__ == '__main__':
    main()
