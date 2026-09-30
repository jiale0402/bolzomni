"""Validation-only diagnosis: native full forward vs sanitized/cached forward."""
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
from poc_model import ROOT, ASSETS, load_model, move, safe_prompt
from generate_panel import decode
from prompt import compile_prompt


def rmsd(a, b):
    a, b = a.double(), b.double()
    a, b = a-a.mean(0), b-b.mean(0)
    u, _, vh = torch.linalg.svd(a.T @ b)
    fix = torch.eye(3, device=a.device, dtype=a.dtype)
    fix[-1,-1] = torch.det(u @ vh)
    return float((((a @ (u @ fix @ vh))-b)**2).sum(-1).mean().sqrt())


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    model = load_model()
    print('DISTANCE_CONDITIONING_ENABLED', model.use_token_distances, flush=True)
    rows = json.loads((ROOT/'manifests/data.json').read_text())['validation'][:2]
    out = ROOT/'experiments/omni_joint/results'
    out.mkdir(parents=True, exist_ok=True)
    report = []
    canonicals = load_canonicals(str(ASSETS/'mols.zip'))
    original_sample = model.structure_module.sample
    for row in rows:
        random.seed(1729); np.random.seed(1729); torch.manual_seed(1729)
        dataset = PredictionDataset(
            Dataset(str(Path(row['directory'])/'fixed.yaml'), Tokenizer(atomize_modified_residues=False), Featurizer()),
            canonicals, str(ASSETS/'mols.zip'), atom14=True, design=True, disulfide_on=False)
        f = move(collate([dataset[0]]), 'cuda')
        truth = torch.bmm(f['token_to_rep_atom'].float(), f['coords'].reshape(1,-1,3))[0]
        dm = f['design_mask'][0].bool()
        outputs = {}
        for mode in ('native', 'sanitized', 'repaired'):
            prompt = model.masker(f) if mode == 'native' else (
                compile_prompt(model,f) if mode == 'repaired' else safe_prompt(model, f, 'fixed'))
            captured = {}
            def sample_at_fixed_seed(**kwargs):
                torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
                captured.update(kwargs)
                return original_sample(**kwargs)
            model.structure_module.sample = sample_at_fixed_seed
            started = time.time()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(prompt, recycling_steps=3, num_sampling_steps=500,
                               diffusion_samples=1, step_scale=2.0, noise_scale=0.88)
            model.structure_module.sample = original_sample
            x = output['sample_atom_coords']
            seq, ca, _ = decode(x, prompt)
            outputs[mode] = x
            record = dict(id=row['id'], mode=mode, sequence=seq,
                          scaffold_rmsd=rmsd(ca[~dm], truth[~dm]), seconds=time.time()-started)
            if mode == 'sanitized':
                torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
                with torch.no_grad():
                    cached = original_sample(**captured)['sample_atom_coords']
                record['cached_max_difference'] = float((cached-x).abs().max())
                record['native_sanitized_max_difference'] = float((outputs['native']-x).abs().max())
            if mode == 'repaired':
                record['native_repaired_max_difference'] = float((outputs['native']-x).abs().max())
                assert record['native_repaired_max_difference']<1e-5
            report.append(record)
            print(json.dumps(record), flush=True)
            (out/'native_diagnostic.json').write_text(json.dumps(report, indent=2))
    print('AUDIT_FINISHED', flush=True)


if __name__ == '__main__':
    main()
