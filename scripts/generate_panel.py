"""Paired native/adapted raw generation, before any sequence replacement or repair."""
import argparse
import json
import time

import numpy as np
import torch
from boltzgen.data import const
from poc_model import ROOT, Feedback, load_model, move, sample, AA_IDS


def decode(coords, feats):
    """Same geometric identity rule as native res_from_atom14 (UNK stays invalid)."""
    design = feats['design_mask'][0].bool()
    atom_to_token = feats['atom_to_token'][0].long().argmax(-1)
    atom_mask = design[atom_to_token] & feats['atom_pad_mask'][0].bool()
    designed = coords[0, atom_mask].reshape(-1, 14, 3)
    distances = torch.cdist(designed[:, :4], designed[:, 4:])
    values, nearest = distances.min(dim=1)
    nearest[values > 0.5] = -1
    counts = (nearest[..., None] == torch.arange(4, device=coords.device)).sum(1)
    names = [const.placement_count_to_token.get(tuple(c), 'UNK') for c in counts.tolist()]
    ids = feats['res_type'][0].long().argmax(-1).tolist()
    for i, name in zip(design.nonzero().flatten().tolist(), names):
        ids[i] = const.token_ids[name]
    sequence = ''.join(const.prot_token_to_letter.get(const.tokens[i], 'X') for i in ids)
    ca = torch.bmm(feats['token_to_rep_atom'].float(), coords)[0]
    return sequence, ca, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['native', 'feedback'], required=True)
    ap.add_argument('--split', choices=['test', 'validation'], default='test')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--candidates', type=int, default=8)
    ap.add_argument('--steps', type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    model = load_model()
    adapter = None
    if args.method == 'feedback':
        checkpoint = torch.load(ROOT / 'outputs/training/best.pt', map_location='cpu', weights_only=False)
        adapter = Feedback(checkpoint['token_dim'], checkpoint['hidden_dim']).cuda().eval()
        adapter.load_state_dict(checkpoint['adapter'])
    rows = json.loads((ROOT / 'manifests/data.json').read_text())[args.split]
    out = ROOT / 'outputs' / f'generated-{args.split}' / args.method
    out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows[:args.limit or None]):
        for condition in ('fixed', 'flexible'):
            cache = move(torch.load(ROOT / 'cache/features' / f'{row["id"]}-{condition}.pt', weights_only=False), 'cuda')
            assert cache['hidden_feature_audit']
            feats = cache['conditioning']['feats']
            for candidate in range(args.candidates):
                stem = f'{row["id"]}-{condition}-{candidate}'
                if (out / (stem + '.json')).exists():
                    continue
                seed = 100000 + i*100 + candidate
                started = time.time()
                coords, logits = sample(model, adapter, cache, args.steps, seed)
                sequence, ca, names = decode(coords, feats)
                native_ca = torch.bmm(feats['token_to_rep_atom'].float(), cache['target'])[0]
                record = {'id': row['id'], 'condition': condition, 'candidate': candidate, 'method': args.method,
                          'sequence': sequence, 'sequence_valid': 'X' not in sequence,
                          'start': row['start'], 'end': row['end'], 'seed': seed,
                          'sampling_steps': args.steps, 'seconds': time.time()-started,
                          'raw_identity_decode': names, 'sequence_source': 'native_atom14_geometry_decoder',
                          'native_sequence_recovery': sum(a == b for a, b in zip(sequence[row['start']:row['end']+1], row['sequence'][row['start']:row['end']+1]))/(row['end']-row['start']+1)}
                if logits is not None:
                    predicted_names = [const.canonical_tokens[x] for x in logits[0, feats['design_mask'][0].bool()].argmax(-1).tolist()]
                    record['feedback_readout_agreement'] = sum(a == b for a,b in zip(names,predicted_names))/len(names)
                np.savez_compressed(out / (stem + '.npz'), ca=ca.cpu().numpy(), native_ca=native_ca.cpu().numpy(), atom_coords=coords[0].cpu().numpy())
                (out / (stem + '.json')).write_text(json.dumps(record, indent=2))
                print('GENERATED', stem, args.method, record['sequence_valid'], flush=True)
    print('GENERATION_COMPLETE', args.method, flush=True)


if __name__ == '__main__':
    main()
