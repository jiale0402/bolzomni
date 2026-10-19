"""The existing compact, sanitized donor contract; no new feature extraction."""
import torch


def check_features(features, generate_mask, batch_ids):
    if not features:
        raise ValueError('Set nonempty graph-ordered donor features before epsilon')
    if not torch.equal(torch.cat([f['generate_mask'] for f in features]), generate_mask):
        raise ValueError('Donor/native generate masks differ')
    expected = torch.cat([torch.full_like(f['mask'], i, dtype=batch_ids.dtype)
                          for i, f in enumerate(features)])
    if not torch.equal(expected, batch_ids):
        raise ValueError('Donor/native graph ordering differs')
    for f in features:
        mask = f['mask']
        if mask.ndim != 1 or mask.dtype != torch.bool or f['generate_mask'].shape != mask.shape:
            raise ValueError('Expected native one-dimensional boolean coverage masks')
        m = int(mask.sum())
        if f['single'].shape != (m, 384) or f['pair'].shape != (m, m, 128):
            raise ValueError('Expected compact aligned single/pair tensors')


def assert_hidden(H, features):
    if H.ndim != 3 or H.shape[-1] != 512 or len(H) != len(features):
        raise ValueError('Expected native padded B,N,512 scalar states')
    if max(len(f['mask']) for f in features) > H.shape[1]:
        raise ValueError('Native scalar sequence is shorter than donor mapping')
