"""EnzyAdapter stage-1 operator transfer; original Anew remains frozen.

Author CrossModalAdapter and MolEmbedder definitions are unchanged. The host
mapping is the post-spatial-attention feed-forward branch input at all six EPT
layers. The native outer residual skip and native vector branch are retained.
"""
import torch
from torch import nn
from contract import check_features, assert_hidden
from enzy_crossmodal import CrossModalAdapter
from enzy_projector import MolEmbedder


class Fusion(nn.Module):
    def __init__(self, variant='full', width=512, attention_width=169, projector_hidden_width=1029):
        super().__init__()
        if variant != 'full' or width != 512:
            raise ValueError('EnzyAdapter transfer uses native 512-dimensional states')
        self.layers = tuple(range(6))
        self.dimension_bridge = nn.Linear(384, 512)
        self.projector = MolEmbedder()
        # Capacity matching changes widths, not the author's two-layer MLP,
        # activation, LayerNorm or six independent attention injection sites.
        self.projector.node_embedder[0] = nn.Linear(512, projector_hidden_width)
        self.projector.node_embedder[2] = nn.Linear(projector_hidden_width, 512)
        self.adapters = nn.ModuleDict({str(i): CrossModalAdapter(512, 512, attention_width)
                                       for i in self.layers})
        self.output_bridges = nn.ModuleDict({str(i): nn.Linear(attention_width, 512)
                                            for i in self.layers})

    def project(self, features):
        return [self.projector(self.dimension_bridge(f['single'].float())) for f in features]

    def forward(self, H, features, layer, projected=None):
        assert_hidden(H, features)
        projected = self.project(features) if projected is None else projected
        updates = []
        for b, f in enumerate(features):
            n, m = len(f['mask']), len(f['single'])
            h = H[b, :n]
            if not m:
                delta = torch.zeros_like(h)
            else:
                delta, _ = self.adapters[str(layer)](
                    h.unsqueeze(0), projected[b].unsqueeze(0), projected[b].unsqueeze(0),
                    torch.ones((1, n), device=h.device, dtype=torch.bool),
                    torch.ones((1, m), device=h.device, dtype=torch.bool))
                delta = self.output_bridges[str(layer)](delta.squeeze(0))
            updates.append(torch.cat([delta, torch.zeros_like(H[b, n:])], dim=0))
        return torch.stack(updates)


class FusionEpsilon(nn.Module):
    def __init__(self, base, fusion, gamma=1.):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.fusion, self.gamma, self.features = fusion, float(gamma), None
        self._active = False
        self.last_calls = {}

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
        calls, handles = {}, []
        self._active = True
        try:
            projected = self.fusion.project(self.features)
            for layer in self.fusion.layers:
                module = self.base.get_submodule(f'encoder.encoder.layer_{layer}.ffn_layer.sub_layer')
                def inject(module, inputs, layer=layer):
                    h, v = inputs
                    delta = self.fusion(h, self.features, layer, projected)
                    calls[layer] = calls.get(layer, 0) + 1
                    return h + self.gamma * delta, v
                handles.append(module.register_forward_pre_hook(inject))
            result = self.base(*args)
            expected = 2 if conditions is not None and conditions.w is not None else 1
            if any(calls.get(layer) != expected for layer in self.fusion.layers):
                raise RuntimeError('EnzyAdapter did not execute at every native/CFG layer')
            self.last_calls = calls
            return result
        finally:
            for handle in handles:
                handle.remove()
            self._active = False
