"""Frozen BoltzGen with a shared token chemical-feedback residual.

Uses native distance conditioning and native sampler; does not claim hard clamping.
"""
import copy
import json
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from boltzgen.data import const
from boltzgen.model.models.boltz import Boltz
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path('/projects/p53127/anewomni_work/boltz_adapter_20260922/assets')
AA_IDS = [const.token_ids[a] for a in const.canonical_tokens]


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, partial):
        return partial(value.func, *(move(v, device) for v in value.args), **move(value.keywords, device))
    if isinstance(value, list):
        return [move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    return value


def load_model():
    config = OmegaConf.load(ROOT / 'upstream/src/boltzgen/resources/config/design.yaml')
    overrides = OmegaConf.to_container(config.override, resolve=True)
    overrides.pop('validators', None)
    overrides.pop('masker_args', None)
    model = Boltz.load_from_checkpoint(
        str(ASSETS / 'boltzgen1_diverse.ckpt'), map_location='cpu', strict=True,
        weights_only=False, use_ema=False, validators=None,
        masker_args={'mask': True, 'mask_backbone': False},
        predict_args={'recycling_steps': 3, 'sampling_steps': 64, 'diffusion_samples': 1},
        **overrides,
    )
    model.eval().requires_grad_(False).cuda()
    model.use_kernels = False
    return model


def safe_prompt(model, feats, condition):
    """Targets stay in a separate object; native model inputs contain no hidden labels."""
    f = model.masker(feats)
    design = f['design_mask'].bool()
    atom_design = torch.bmm(f['atom_to_token'].float(), design.float().unsqueeze(-1)).squeeze(-1).bool()
    # These are target/metadata fields, not information a prompt may reveal.
    f['res_type_clone'] = f['res_type'].clone()
    for key in ('coords', 'bfactor', 'plddt', 'fake_atom_mask'):
        if key in f:
            f[key] = torch.zeros_like(f[key])
    f['atom_resolved_mask'] = f['atom_pad_mask'].clone()
    f['ref_chirality'][atom_design] = const.chirality_type_ids['CHI_UNSPECIFIED']
    if condition == 'flexible':
        f['center_coords'] = f['center_coords'].masked_fill(design[..., None], 0)
    # No identity-specific occupancy in the design slots.
    assert torch.all(f['atom_pad_mask'][atom_design] == 1)
    return f


class Feedback(nn.Module):
    def __init__(self, token_dim, hidden_dim):
        super().__init__()
        self.token_dim, self.hidden_dim = token_dim, hidden_dim
        self.norm = nn.LayerNorm(token_dim)
        self.residual = nn.Sequential(nn.Linear(token_dim + 24, 128), nn.SiLU(), nn.Linear(128, token_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.readout = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 20))

    def forward(self, s, chemical, design, fixed, log_sigma):
        visibility = torch.stack((design.float(), torch.full_like(design.float(), float(fixed))), dim=-1)
        time = torch.ones_like(design.float())[..., None] * log_sigma.reshape(-1, 1, 1)
        return self.residual(torch.cat((self.norm(s), chemical, visibility, time), dim=-1))


def labels(feats):
    ids = feats['res_type'].argmax(-1)
    table = ids.new_full((const.num_tokens,), -1)
    table[torch.tensor(AA_IDS, device=ids.device)] = torch.arange(20, device=ids.device)
    result = table[ids]
    assert (result >= 0).all(), 'Noncanonical protein outside pilot scope'
    return result


def chemistry(target, design, reveal=0.0):
    state = F.one_hot(target, 21).float()
    hidden = design & (torch.rand_like(design.float()) >= reveal)
    state[hidden] = F.one_hot(torch.tensor(20, device=target.device), 21).float()
    return state, hidden


class LiveReadout:
    def __init__(self, model):
        self.value = None
        self.handle = model.structure_module.score_model.a_norm.register_forward_hook(self.capture)

    def capture(self, module, args, output):
        self.value = output

    def close(self):
        self.handle.remove()


def denoise(model, adapter, live, cache, noisy, sigma, chemical):
    kwargs = dict(cache['conditioning'])
    if adapter is not None:
        kwargs['s_trunk'] = kwargs['s_trunk'] + adapter(
            kwargs['s_trunk'], chemical, kwargs['feats']['design_mask'].bool(),
            cache['condition'] == 'fixed', torch.log(sigma.clamp_min(1e-6)),
        )
    pred, _ = model.structure_module.preconditioned_network_forward(
        noisy, sigma, {**kwargs, 'multiplicity': 1}, training=False,
    )
    logits = adapter.readout(live.value) if adapter is not None else None
    return pred, logits


def sample(model, adapter, cache, steps, seed):
    """Use the unchanged native sampler, injecting feedback once per score evaluation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = cache['conditioning']
    feats = kwargs['feats']
    design = feats['design_mask'].bool()
    # Only observed identities initialize the state; hidden labels are never used.
    initial = feats['res_type'].argmax(-1)
    table = initial.new_full((const.num_tokens,), 20)
    table[torch.tensor(AA_IDS, device=initial.device)] = torch.arange(20, device=initial.device)
    state = F.one_hot(table[initial], 21).float()
    state[design] = F.one_hot(torch.tensor(20, device=initial.device), 21).float()
    original = model.structure_module.score_model.forward
    live = LiveReadout(model)
    calls = 0
    final_logits = None
    chem_rng = torch.Generator(device=initial.device).manual_seed(seed + 1777)

    def wrapped(*args, **kw):
        nonlocal calls, state, final_logits
        kw = dict(kw)
        if adapter is not None:
            # Native times = log(sigma/sigma_data)/4.
            log_sigma = kw['times'] * 4 + torch.log(kw['times'].new_tensor(model.structure_module.sigma_data))
            kw['s_trunk'] = kw['s_trunk'] + adapter(kw['s_trunk'], state, design, cache['condition'] == 'fixed', log_sigma)
        out = original(*args, **kw)
        calls += 1
        if adapter is not None:
            final_logits = adapter.readout(live.value)
            hidden = design & (state.argmax(-1) == 20)
            reveal = hidden & (torch.rand(hidden.shape, device=hidden.device, generator=chem_rng) < 1 / max(1, steps - calls + 1))
            sampled = torch.multinomial(final_logits.softmax(-1).reshape(-1, 20), 1, generator=chem_rng).reshape(hidden.shape)
            state = torch.where(reveal[..., None], F.one_hot(sampled, 21).float(), state)
        return out

    model.structure_module.score_model.forward = wrapped
    try:
        with torch.no_grad():
            out = model.structure_module.sample(
                **kwargs, atom_mask=feats['atom_pad_mask'].float(), multiplicity=1,
                num_sampling_steps=steps, step_scale=2.0, noise_scale=0.88,
                inference_logging=False,
            )
    finally:
        model.structure_module.score_model.forward = original
        live.close()
    assert calls == steps, (calls, steps)
    return out['sample_atom_coords'], final_logits
