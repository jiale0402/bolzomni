"""Untrained draft: expose cached donor information to native EPT attention.

No diffusion, transition, loss, decoder, or native conditioning code is replaced.
The adapter adds an invariant logit bias; native attention applies it to both H
and V. A donor-covered (i,j) retains its pair feature. An uncovered query (e.g.
a generated ligand node) can read covered donor keys using its current H.
Unknown donor pairs remain absent, rather than being fabricated.
"""
import math
from collections import Counter

import torch
from torch import nn


class DonorAttentionBias(nn.Module):
    def __init__(self, hidden_size, n_heads, layers, rank_per_head=8, pair_width=32):
        super().__init__()
        self.layers = tuple(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError('Select explicit, unique native EPT layer indices')
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.rank = rank_per_head
        self.h_norm = nn.LayerNorm(hidden_size)
        self.s_norm = nn.LayerNorm(384)
        self.pair_encoder = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, pair_width), nn.SiLU())
        self.query = nn.ModuleDict()
        self.key = nn.ModuleDict()
        self.pair_out = nn.ModuleDict()
        for layer in self.layers:
            name = str(layer)
            self.query[name] = nn.Linear(hidden_size, n_heads * rank_per_head, bias=False)
            self.key[name] = nn.Linear(384, n_heads * rank_per_head, bias=False)
            self.pair_out[name] = nn.Linear(pair_width, n_heads, bias=False)
            # Exact native output at initialization. Gradients reach these two
            # output projections immediately; earlier weights start next update.
            nn.init.zeros_(self.key[name].weight)
            nn.init.zeros_(self.pair_out[name].weight)

    def forward(self, H, valid_nodes, features, layer):
        if layer not in self.layers or H.shape[-1] != self.hidden_size:
            raise ValueError('Native layer/hidden dimension does not match adapter')
        if valid_nodes.shape != H.shape[:2] or valid_nodes.dtype != torch.bool:
            raise ValueError('Expected native boolean node mask')
        if len(features) != len(H):
            raise ValueError('Exactly one donor record per native graph is required')
        name = str(layer)
        bias = H.new_zeros((len(H), self.n_heads, H.shape[1], H.shape[1]))
        for b, f in enumerate(features):
            positions = valid_nodes[b].nonzero(as_tuple=True)[0]
            mask = f['mask']
            if mask.dtype != torch.bool or mask.ndim != 1 or len(mask) != len(positions):
                raise ValueError('Donor mask must match native graph nodes exactly')
            donor_local = mask.nonzero(as_tuple=True)[0]
            donor_positions = positions[donor_local]
            m = len(donor_local)
            if f['single'].shape != (m, 384) or f['pair'].shape != (m, m, 128):
                raise ValueError('Compact single/pair dimensions do not match donor mask')
            if not m:
                continue
            param_dtype = self.h_norm.weight.dtype
            # Every real native node can query existing donor keys. No ligand
            # donor feature, ligand identity, or target coordinate is invented.
            q = self.query[name](self.h_norm(H[b, positions].to(param_dtype)))
            k = self.key[name](self.s_norm(f['single'].to(param_dtype)))
            q = q.reshape(len(positions), self.n_heads, self.rank)
            k = k.reshape(m, self.n_heads, self.rank)
            cross = torch.einsum('ihd,jhd->hij', q, k) / math.sqrt(self.rank)
            bias[b, :, positions[:, None], donor_positions[None, :]] = cross.to(H.dtype)
            # Keep both i and j: no reduction of donor z_ij into a token vector.
            # Chunk intermediate activations; only the small head-wise bias is dense.
            for start in range(0, m, 32):
                stop = min(start + 32, m)
                pair = self.pair_encoder(f['pair'][start:stop].to(param_dtype))
                pair = self.pair_out[name](pair).permute(2, 0, 1).to(H.dtype)
                rows = donor_positions[start:stop]
                bias[b, :, rows[:, None], donor_positions[None, :]] = (
                    bias[b, :, rows[:, None], donor_positions[None, :]] + pair
                )
        return bias


class PairBiasEpsilon(nn.Module):
    """Wrap a plain native EpsilonNet, never an existing FusionEpsilon wrapper.

    Hooks are scoped to one synchronous epsilon call and removed on exceptions.
    This draft deliberately rejects concurrent/reentrant calls and checkpointed
    backward re-entry. Production checkpointing needs a persistent explicit API.
    """
    def __init__(self, base, adapter, gamma=1.):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.fusion = adapter
        self.gamma = float(gamma)
        self.features = None
        self._active = False
        self.last_calls = {}
        encoder = self.base.encoder.encoder
        self._paths = {}
        for layer in adapter.layers:
            if not (0 <= layer < encoder.n_layers):
                raise ValueError(f'Layer {layer} is absent from the actual native encoder')
            module = encoder.get_submodule(f'layer_{layer}.attn_layer.sub_layer')
            if (module.layer_idx, module.d_hidden, module.n_heads) != (
                layer, adapter.hidden_size, adapter.n_heads
            ):
                raise ValueError('Native EPT attention dimensions do not match adapter')
            self._paths[layer] = f'encoder.encoder.layer_{layer}.attn_layer.sub_layer'

    @classmethod
    def for_native(cls, base, layers, gamma=1., **adapter_options):
        encoder = base.encoder.encoder
        first = encoder.get_submodule(f'layer_{layers[0]}.attn_layer.sub_layer')
        adapter = DonorAttentionBias(first.d_hidden, first.n_heads, layers, **adapter_options)
        adapter.to(device=first.scaler_q.weight.device, dtype=first.scaler_q.weight.dtype)
        return cls(base, adapter, gamma)

    @property
    def require_block_edges(self):
        return self.base.require_block_edges

    def set_features(self, features, cache=False):
        # No persistent projections or graph tensors survive an optimizer update.
        self.features = features

    def forward(self, H, X, c, edges, edge_types, generate_mask, batch_ids, beta, conditions=None):
        args = (H, X, c, edges, edge_types, generate_mask, batch_ids, beta, conditions)
        if self.gamma == 0:
            return self.base(*args)
        if self._active:
            raise RuntimeError('Concurrent/reentrant epsilon calls are unsupported')
        if self.features is None:
            raise ValueError('Call set_features with the matching batch first')
        expected_ids = torch.cat([
            torch.full_like(f['mask'], b, dtype=batch_ids.dtype) for b, f in enumerate(self.features)
        ])
        if not torch.equal(batch_ids, expected_ids):
            raise ValueError('Native batch order and donor records differ')
        if not torch.equal(generate_mask, torch.cat([f['generate_mask'] for f in self.features])):
            raise ValueError('Native generation mask and donor records differ')
        calls, handles = Counter(), []
        self._active = True
        try:
            for layer, path in self._paths.items():
                def inject(module, inputs, kwargs, layer=layer):
                    H_layer = inputs[0]
                    if 'cached_info' in kwargs:
                        cached = kwargs['cached_info']
                    elif len(inputs) >= 3:
                        cached = inputs[2]
                    else:
                        raise ValueError('Native EPT cached_info is missing')
                    D, rbf, valid_nodes = cached
                    delta = self.fusion(H_layer, valid_nodes, self.features, layer)
                    # The real SelfAttnLayer indexes just its own entry. A tuple
                    # lets that one entry have per-head bias without copying all
                    # other layers or mutating the shared native rbf tensor.
                    replacement = list(rbf)
                    replacement[layer] = rbf[layer] + self.gamma * delta
                    changed = (D, tuple(replacement), valid_nodes)
                    calls[layer] += 1
                    if 'cached_info' in kwargs:
                        return inputs, {**kwargs, 'cached_info': changed}
                    return (*inputs[:2], changed, *inputs[3:]), kwargs
                handles.append(self.base.get_submodule(path).register_forward_pre_hook(inject, with_kwargs=True))
            outputs = self.base(*args)
            expected = 2 if conditions is not None and conditions.w is not None else 1
            if any(calls[layer] != expected for layer in self.fusion.layers):
                raise RuntimeError(f'Unexpected native injection calls: {dict(calls)}')
            self.last_calls = dict(calls)
            return outputs
        finally:
            for handle in handles:
                handle.remove()
            self._active = False
