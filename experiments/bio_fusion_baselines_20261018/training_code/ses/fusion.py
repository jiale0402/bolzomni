"""SES-Adapter fusion at the first normalized EPT feed-forward branch input.

Continuous donor input projection replaces discrete structure-token embedding.
Preserve author Q/K/V direction, RoPE, dropout, residual addition and LayerNorm.
"""
from types import SimpleNamespace
import torch
from torch import nn
from contract import check_features
from ses_crossmodal import CrossModalAttention


class Fusion(nn.Module):
    def __init__(self, variant='full', width=512, hidden_width=944):
        super().__init__()
        if variant != 'full' or width != 512:
            raise ValueError('SES-Adapter transfer uses native 512-dimensional states')
        if hidden_width % 16:
            raise ValueError('Eight RoPE heads require an even head dimension')
        self.structure_projection = nn.Linear(384, hidden_width)
        self.native_projection = nn.Linear(512, hidden_width)
        self.cross_attention = CrossModalAttention(SimpleNamespace(
            hidden_size=hidden_width, num_attention_heads=8, attention_probs_dropout_prob=0.0))
        self.layer_norm = nn.LayerNorm(hidden_width)
        self.output_bridge = nn.Linear(hidden_width, 512)
        # The pinned author forward never applies out_proj. Preserve that fact;
        # do not train parameters that have no path to any prediction.
        self.cross_attention.out_proj.requires_grad_(False)

    def forward(self, c, features):
        offset, updates = 0, []
        for f in features:
            n = len(f['mask'])
            h = c[offset:offset+n]
            offset += n
            idx = f['mask'].nonzero(as_tuple=True)[0]
            if not len(idx):
                updates.append(torch.zeros_like(h))
                continue
            # Full native positions, including coverage gaps, enter RoPE. Padded
            # query outputs are discarded; no missing donor information is used.
            projected_h = self.native_projection(h)
            q = torch.zeros_like(projected_h).index_copy(0, idx, self.structure_projection(f['single'].float()))
            context = self.cross_attention(q.unsqueeze(0), projected_h.unsqueeze(0), projected_h.unsqueeze(0),
                torch.ones((1, n), dtype=torch.bool, device=c.device)).squeeze(0)
            fused = self.layer_norm(projected_h + context)
            delta = self.output_bridge((fused-projected_h)[idx])
            updates.append(torch.zeros_like(h).index_copy(0, idx, delta))
        if offset != len(c):
            raise ValueError('Donor features do not partition native conditioning')
        return torch.cat(updates)


class FusionEpsilon(nn.Module):
    def __init__(self, base, fusion, gamma=1.):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.fusion, self.gamma, self.features = fusion, float(gamma), None
        self._active = False

    @property
    def require_block_edges(self):
        return self.base.require_block_edges

    def set_features(self, features, cache=False):
        self.features = features

    def forward(self, H, X, c, edges, edge_types, generate_mask, batch_ids, beta, conditions=None):
        args = (H, X, c, edges, edge_types, generate_mask, batch_ids, beta, conditions)
        if self.gamma == 0:
            return self.base(*args)
        check_features(self.features, generate_mask, batch_ids)
        if self._active:
            raise RuntimeError('Reentrant native epsilon calls are unsupported')
        # Transfer the author's normalized scalar-feature operator once at the
        # first native post-attention FFN input. Native pre-norm/outer residual
        # and vector input stay intact. Later frozen EPT blocks propagate the
        # covered context update to generated nodes (whose donor is unknown).
        calls = 0
        def fuse(module, inputs):
            nonlocal calls
            scalar, vector = inputs
            calls += 1
            flat = torch.cat([scalar[b, :len(f['mask'])] for b, f in enumerate(self.features)])
            delta = self.fusion(flat, self.features)
            offset, padded = 0, []
            for b, f in enumerate(self.features):
                n = len(f['mask'])
                padded.append(torch.cat([delta[offset:offset+n], torch.zeros_like(scalar[b, n:])]))
                offset += n
            return scalar + self.gamma * torch.stack(padded), vector
        self._active = True
        branch = self.base.get_submodule('encoder.encoder.layer_0.ffn_layer.sub_layer')
        handle = branch.register_forward_pre_hook(fuse)
        try:
            result = self.base(*args)
            expected = 2 if conditions is not None and conditions.w is not None else 1
            if calls != expected:
                raise RuntimeError('SES-Adapter did not execute at each native/CFG first FFN input')
            return result
        finally:
            handle.remove()
            self._active = False
