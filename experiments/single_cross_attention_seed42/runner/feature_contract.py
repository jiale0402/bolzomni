"""Rebind context-only ligand donor features to independently sampled sizes."""
import torch


def context_positions(data):
    desc = data['cplx_desc']
    positions = desc.pocket_block_ids + desc.lig_block_ids
    assert len(positions) == len(data['generate_mask'])
    return [(c, tuple(r)) for (c, r), g in
            zip(positions, data['generate_mask'].tolist()) if not g]


def bind_features(feature, cached, current, benchmark):
    assert feature['compact']
    assert torch.equal(feature['generate_mask'], cached['generate_mask'])
    assert len(feature['mask']) == len(cached['S'])
    if benchmark != 'crossdocked2020':
        assert torch.equal(cached['generate_mask'], current['generate_mask'])
        for field in ['S', 'A', 'X', 'position_ids', 'block_lengths', 'bonds']:
            torch.testing.assert_close(cached[field], current[field], rtol=0, atol=0)
        return {k: feature[k] for k in ['single', 'pair', 'mask', 'generate_mask']}
    assert context_positions(cached) == context_positions(current)
    old_mask, new_mask = ~cached['generate_mask'], ~current['generate_mask']
    assert not feature['mask'][cached['generate_mask']].any(), 'Ligand donor covers generated blocks'
    for field in ['S', 'position_ids', 'is_aa', 'block_lengths']:
        torch.testing.assert_close(cached[field][old_mask], current[field][new_mask], rtol=0, atol=0)
    for data, mask in [(cached, old_mask), (current, new_mask)]:
        assert int(data['block_lengths'].sum()) == len(data['A'])
    old_atoms = torch.repeat_interleave(old_mask, cached['block_lengths'])
    new_atoms = torch.repeat_interleave(new_mask, current['block_lengths'])
    for field in ['X', 'A']:
        torch.testing.assert_close(cached[field][old_atoms], current[field][new_atoms], rtol=0, atol=0)
    coverage = torch.zeros_like(new_mask)
    coverage[new_mask] = feature['mask'][old_mask]
    count = int(coverage.sum())
    assert feature['single'].shape[0] == count
    assert feature['pair'].shape[:2] == (count, count)
    return {'single': feature['single'], 'pair': feature['pair'],
            'mask': coverage, 'generate_mask': current['generate_mask'].clone()}
