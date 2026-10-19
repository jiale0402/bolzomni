import functools
import sys
import torch
from common import ROOT, SOURCE, CONFIG, load_official
sys.path.insert(0, str(SOURCE))
from data.base import BaseDataset
from fusion import Fusion, FusionEpsilon

def materialize(rows):
    assert all('variants' not in r for r in rows)
    return list(rows)

def load_input(row):
    return torch.load(row.get('input_path', str(ROOT/'inputs'/row['key']/'anew.pt')),
                      map_location='cpu', weights_only=True)

@functools.lru_cache(maxsize=32)
def cpu_feature(path):
    f = torch.load(path, map_location='cpu', weights_only=True)
    if not f.get('compact', False):
        mask = f['mask'].bool()
        f['single'] = f['single'][mask]
        f['pair'] = f['pair'][mask][:, mask]
    return {k: f[k] for k in ('single', 'pair', 'mask', 'generate_mask')}

def features(rows, device='cuda'):
    return [{k: v.to(device) for k, v in cpu_feature(r.get('feature_path',
        str(ROOT/'features/aligned'/(r['key']+'.pt')))).items()} for r in rows]

def native_batch(rows, device='cuda'):
    batch = BaseDataset.collate_fn(None, [load_input(r) for r in rows])
    return {k: v.to(device) for k, v in batch.items()}

def build_model():
    model, _ = load_official()
    model.eval().requires_grad_(False).cuda()
    if CONFIG['train_backbone']:
        model.diffusion.eps_net.requires_grad_(True)
    frozen = [p for p in model.parameters() if not p.requires_grad]
    fusion = Fusion(**CONFIG.get('fusion_kwargs', {})).cuda()
    count = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    assert count == CONFIG['trainable_parameters'], (count, CONFIG['trainable_parameters'])
    assert abs(count / CONFIG['parameter_match_target'] - 1) <= CONFIG['parameter_match_relative_tolerance']
    epsilon = FusionEpsilon(model.diffusion.eps_net, fusion).cuda()
    model.diffusion.eps_net = epsilon
    return model, fusion, epsilon, frozen

def microbatches(rows, max_cost=140000, max_count=4):
    group, cost = [], 0
    for r in rows:
        new = r['nblocks']**2
        if group and (cost+new > max_cost or len(group) >= max_count):
            yield group
            group, cost = [], 0
        group.append(r)
        cost += new
    if group:
        yield group
