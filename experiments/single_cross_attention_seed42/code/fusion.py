"""One conditioning cross-attention, widened from the historical full adapter.

Only projection widths change: c <- c + gamma * cross_attention(c, single, pair).
Native epsilon, all EPT layers, CFG, diffusion, losses and decoding stay native.
Uncovered nodes receive no direct update, as in the historical implementation.
"""
import math
import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self, variant='full', width=2040, inner=768, heads=4, query_chunk=16):
        super().__init__()
        if variant != 'full' or inner % heads or min(width, inner, heads, query_chunk) < 1:
            raise ValueError('Expected full donor features and positive, head-divisible widths')
        self.inner, self.heads, self.query_chunk = inner, heads, query_chunk
        self.single_mlp = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, width), nn.SiLU(), nn.Linear(width, 512))
        self.pair_mlp = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, width), nn.SiLU(), nn.Linear(width, 128))
        self.query = nn.Linear(512, inner, bias=False)
        self.key = nn.Linear(512, inner, bias=False)
        self.value = nn.Linear(512, inner, bias=False)
        self.pair_bias = nn.Linear(128, heads, bias=False)
        self.pair_value = nn.Linear(128, inner, bias=False)
        self.output = nn.Linear(inner, 512)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, c, features):
        offset, result = 0, []
        depth = self.inner // self.heads
        for f in features:
            mask = f['mask']
            if mask.ndim != 1 or mask.dtype != torch.bool:
                raise ValueError('Expected a one-dimensional boolean coverage mask')
            n = len(mask)
            ci = c[offset:offset+n]
            offset += n
            idx = mask.nonzero(as_tuple=True)[0]
            m = len(idx)
            if f['single'].shape != (m, 384) or f['pair'].shape != (m, m, 128):
                raise ValueError('Expected compact, aligned single/pair donor features')
            if not m:
                result.append(torch.zeros_like(ci))
                continue
            s = self.single_mlp(f['single'].float())
            q = self.query(ci[idx]).reshape(m, self.heads, depth)
            k = self.key(s).reshape(m, self.heads, depth)
            v = self.value(s).reshape(m, self.heads, depth)
            chunks = []
            for start in range(0, m, self.query_chunk):
                stop = min(start+self.query_chunk, m)
                p = self.pair_mlp(f['pair'][start:stop].float())
                logits = torch.einsum('qhd,khd->hqk', q[start:stop], k)/math.sqrt(depth)
                weights = (logits+self.pair_bias(p).permute(2, 0, 1)).softmax(-1)
                value = torch.einsum('hqk,khd->qhd', weights, v)
                value = value+torch.einsum('hqk,qkhd->qhd', weights,
                    self.pair_value(p).reshape(stop-start, m, self.heads, depth))
                chunks.append(self.output(value.flatten(1)))
            result.append(torch.zeros_like(ci).index_copy(0, idx, torch.cat(chunks)))
        if offset != len(c) or not result:
            raise ValueError('Features must partition all native conditioning nodes')
        return torch.cat(result)


class FusionEpsilon(nn.Module):
    def __init__(self, base, fusion, gamma=1.):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.fusion, self.gamma, self.features = fusion, float(gamma), None

    @property
    def require_block_edges(self):
        return self.base.require_block_edges

    def set_features(self, features, cache=False):
        # This baseline deliberately recomputes its single attention each call.
        self.features = features

    def forward(self, H, X, c, edges, edge_types, generate_mask, batch_ids, beta, conditions=None):
        args = (edges, edge_types, generate_mask, batch_ids, beta, conditions)
        if self.gamma == 0:
            return self.base(H, X, c, *args)
        if self.features is None:
            raise ValueError('Set donor features before calling epsilon')
        if not torch.equal(torch.cat([f['generate_mask'] for f in self.features]), generate_mask):
            raise ValueError('Generated-node masks differ from native input')
        expected_ids = torch.cat([torch.full_like(f['mask'], i, dtype=batch_ids.dtype)
                                  for i, f in enumerate(self.features)])
        if not torch.equal(batch_ids, expected_ids):
            raise ValueError('Donor graph order differs from native input')
        return self.base(H, X, c+self.gamma*self.fusion(c, self.features), *args)
