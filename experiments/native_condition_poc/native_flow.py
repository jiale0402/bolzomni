"""Omni feedback inside the unmodified native sampler, for native relative prompts.

This PoC does not implement user-specified absolute-coordinate observations. Native
structure groups stay in the native feature tensors, rather than becoming hard anchors.
"""
import torch
from full_model import FullOmni as OriginalFullOmni, gradient_norm
from corrected_model import refresh_teacher_bridge, augmented_target
from model import Observations


def native_observations(payload):
    """Convert the documented legacy relative-group cache without changing that file."""
    obs = payload['observations']
    feats = payload['cache']['conditioning']['feats']
    relative_tokens = feats['token_distance_mask'].bool().any(-1)
    relative_atoms = torch.bmm(feats['atom_to_token'].float(), relative_tokens.float()[..., None]).squeeze(-1).bool()
    expected = relative_atoms & feats['backbone_mask'].bool() & payload['resolved'].bool()
    # The old cache's mask must be exactly the derived relative-group mask. Refuse
    # to discard a genuine/unknown absolute observation silently.
    assert torch.equal(obs['coordinate_mask'], expected), 'Unrecognized observation semantics'
    return dict(payload, observations=dict(identity=obs['identity'], identity_mask=obs['identity_mask'],
        coordinates=torch.zeros_like(obs['coordinates']), coordinate_mask=torch.zeros_like(expected)),
        observation_semantics='native relative structure groups; no absolute observations')


class FullOmni(OriginalFullOmni):
    def __init__(self, base, teacher, cache):
        super().__init__(base, teacher, cache)
        # The sampler temporarily intercepts the public entry point. Denoising
        # must call this saved native function, avoiding recursive interception.
        object.__setattr__(self, '_native_forward', base.structure_module.preconditioned_network_forward)

    def denoise(self, cache, noisy, sigma, state, observations, chemical_t, native=False):
        assert not observations.coordinate_mask.any(), 'Absolute anchors are outside this PoC'
        sigma = torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32).reshape(-1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.training):
            if native:
                pred, _ = self.teacher.structure_module.preconditioned_network_forward(
                    noisy, sigma, dict(cache['conditioning'], multiplicity=1), training=False)
                return pred, None
            kwargs = dict(cache['conditioning'])
            ds, dz = self.adapter(kwargs['s_trunk'], cache['z_trunk'], state, observations,
                kwargs['feats']['atom_to_token'], sigma, chemical_t)
            s = kwargs['s_trunk'] + ds
            with torch.autocast('cuda', dtype=torch.bfloat16):
                q, c, keys, enc, dec, token = self.base.diffusion_conditioning(
                    s_trunk=s.to(cache['bridge_s'].dtype),
                    z_trunk=(cache['z_trunk'] + dz).to(cache['z_trunk'].dtype),
                    relative_position_encoding=cache['relative_position_encoding'], feats=kwargs['feats'])
            kwargs['s_trunk'] = s
            kwargs['diffusion_conditioning'] = dict(q=q, c=c, to_keys=keys,
                atom_enc_bias=enc, atom_dec_bias=dec, token_trans_bias=token)
            pred, _ = self._native_forward(noisy, sigma, dict(kwargs, multiplicity=1), training=False)
            return pred, self.adapter.identity_readout(self.live)


@torch.no_grad()
def sample(model, cache, observations, steps=500, seed=1729, step_scale=2., noise_scale=.88):
    observations.validate()
    assert not observations.coordinate_mask.any()
    assert observations.identity.shape[0] == 1
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    state = observations.identity.clone()
    rng = torch.Generator(device=state.device).manual_seed(seed + 1777)
    sm = model.base.structure_module
    original = sm.preconditioned_network_forward
    calls, logits = 0, None

    def feedback_forward(noisy, sigma, network_condition_kwargs, training=False):
        nonlocal calls, logits, state
        assert not training and calls < steps
        chemical_t = noisy.new_tensor([(steps - calls) / steps])
        prediction, logits = model.denoise(cache, noisy, sigma, state, observations, chemical_t)
        reveal = (state == model.adapter.vocabulary) & (~observations.identity_mask)
        reveal &= torch.rand(state.shape, device=state.device, generator=rng) < 1 / (steps - calls)
        proposed = torch.multinomial(logits.float().softmax(-1).reshape(-1, model.adapter.vocabulary),
            1, generator=rng).reshape(state.shape)
        state = torch.where(reveal, proposed, state)
        state = torch.where(observations.identity_mask, observations.identity, state)
        calls += 1
        return prediction, {}

    sm.preconditioned_network_forward = feedback_forward
    try:
        output = sm.sample(**cache['conditioning'], atom_mask=cache['conditioning']['feats']['atom_pad_mask'].float(),
            multiplicity=1, num_sampling_steps=steps, step_scale=step_scale, noise_scale=noise_scale)
    finally:
        sm.preconditioned_network_forward = original
    assert calls == steps and not (state == model.adapter.vocabulary).any()
    return dict(coordinates=output['sample_atom_coords'], identity=state, identity_logits=logits,
        feedback_calls=calls, native_sampler=True)


def losses(model, payload, seed=None, chemical_t_override=None, sigma_override=None):
    """Same joint conditional objective for both observation patterns.

    All coordinates get native noise. Both coordinate objectives use native sigma
    scaling. Relative motifs are conditioned in features, never projected into x.
    """
    from torch.nn import functional as F
    from boltzgen.model.loss.diffusion import weighted_rigid_align, smooth_lddt_loss
    if seed is not None: torch.manual_seed(seed)
    target, obs = augmented_target(payload)
    obs.validate(); assert not obs.coordinate_mask.any()
    cache = payload['cache']; labels = payload['labels']; resolved = payload['resolved'].float()
    sm = model.base.structure_module
    sigma = sm.noise_distribution(len(target)) if sigma_override is None else target.new_full((len(target),), sigma_override)
    t = torch.rand(len(target), device=target.device)
    choice = torch.rand_like(t)
    t = torch.where(choice < .25, torch.ones_like(t), t)
    t = torch.where(choice > .75, torch.zeros_like(t), t)
    if chemical_t_override is not None: t = torch.full_like(t, chemical_t_override)
    hidden = (~obs.identity_mask) & (torch.rand_like(labels.float()) < t[:, None])
    state = torch.where(hidden, model.adapter.vocabulary, labels)
    state = torch.where(obs.identity_mask, obs.identity, state)
    noisy = target + sigma[:, None, None] * torch.randn_like(target)
    pred, logits = model.denoise(cache, noisy, sigma, state, obs, t)
    with torch.no_grad():
        aligned = weighted_rigid_align(target, pred.detach().float(), torch.ones_like(resolved), resolved)
    weight = sm.loss_weight(sigma)
    coordinate = ((((pred.float() - aligned)**2).sum(-1) * resolved).sum(-1)
        / (3 * resolved.sum(-1).clamp_min(1)) * weight).mean()
    spatial = smooth_lddt_loss(pred.float(), target, torch.zeros_like(resolved), resolved, multiplicity=1)
    identity = F.cross_entropy(logits[hidden].float(), labels[hidden]) if hidden.any() else logits.sum() * 0
    native_state, native_t = obs.identity.clone(), torch.ones_like(t)
    with torch.no_grad():
        teacher, _ = model.denoise(cache, noisy, sigma, native_state, obs, native_t, native=True)
    if torch.equal(state, native_state) and torch.equal(t, native_t): student = pred
    else: student, _ = model.denoise(cache, noisy, sigma, native_state, obs, native_t)
    replay_raw = ((((student.float() - teacher.float())**2).sum(-1) * resolved).sum(-1)
        / (3 * resolved.sum(-1).clamp_min(1)))
    replay = (replay_raw * weight).mean()
    return dict(coordinate=coordinate, smooth_lddt=spatial, identity=identity, replay=replay), dict(
        sigma=float(sigma.mean()), chemical_t=float(t.mean()), masked_identities=int(hidden.sum()),
        replay_unweighted=float(replay_raw.detach().mean()))
