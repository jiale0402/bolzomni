"""Candidate noise-matched chemistry with the original coordinate solver intact.

Uses the completed native-condition model/losses as imports; never edits them.
No model execution has validated this candidate until a separately authorized run.
"""
import math
import torch
from native_flow import FullOmni, losses, native_observations, refresh_teacher_bridge
from chemical_schedule import chemical_levels, reveal_probabilities


def sampler_chemistry(structure_module, steps):
    if structure_module.sampling_schedule == 'dilated':
        sigmas = structure_module.sample_schedule_dilated(steps)
    elif structure_module.sampling_schedule == 'af3':
        sigmas = structure_module.sample_schedule_af3(steps)
    else:
        raise ValueError('Unsupported native schedule')
    base = sigmas.detach().cpu().tolist()
    assert len(base) == steps + 1 and base[-1] == 0
    actual = [s * (1 + (structure_module.gamma_0 if nxt > structure_module.gamma_min else 0))
              for s,nxt in zip(base,base[1:])]
    levels = chemical_levels(actual, sigma_data=structure_module.sigma_data,
        p_mean=structure_module.P_mean, p_std=structure_module.P_std)
    return actual, levels, reveal_probabilities(levels)


@torch.no_grad()
def sample(model, cache, observations, steps=500, seed=1729, step_scale=2., noise_scale=.88,
           trace_callback=None):
    observations.validate()
    assert not observations.coordinate_mask.any(), 'Preserve native relative-coordinate semantics'
    assert observations.identity.shape[0] == 1
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    state = observations.identity.clone()
    rng = torch.Generator(device=state.device).manual_seed(seed + 1777)
    sm = model.base.structure_module
    sigmas, levels, reveal_p = sampler_chemistry(sm, steps)
    original = sm.preconditioned_network_forward
    calls, logits = 0, None

    def feedback_forward(noisy, sigma, network_condition_kwargs, training=False):
        nonlocal calls, logits, state
        assert not training and calls < steps
        actual_sigma = float(torch.as_tensor(sigma).reshape(-1)[0])
        assert math.isclose(actual_sigma, sigmas[calls], rel_tol=1e-6, abs_tol=1e-9), 'Coordinate schedule drift'
        chemical_t = noisy.new_tensor([levels[calls]])
        prediction, logits = model.denoise(cache, noisy, sigma, state, observations, chemical_t)
        reveal = (state == model.adapter.vocabulary) & (~observations.identity_mask)
        reveal &= torch.rand(state.shape, device=state.device, generator=rng) < reveal_p[calls]
        proposed = torch.multinomial(logits.float().softmax(-1).reshape(-1, model.adapter.vocabulary),
            1, generator=rng).reshape(state.shape)
        state = torch.where(reveal, proposed, state)
        state = torch.where(observations.identity_mask, observations.identity, state)
        if trace_callback is not None:
            trace_callback(dict(call=calls, sigma=actual_sigma, chemical_t=levels[calls],
                next_chemical_t=levels[calls+1], reveal_probability=reveal_p[calls],
                state=state, logits=logits, denoised=prediction))
        calls += 1
        return prediction, {}

    sm.preconditioned_network_forward = feedback_forward
    try:
        output = sm.sample(**cache['conditioning'], atom_mask=cache['conditioning']['feats']['atom_pad_mask'].float(),
            multiplicity=1, num_sampling_steps=steps, step_scale=step_scale, noise_scale=noise_scale)
    finally:
        sm.preconditioned_network_forward = original
    assert calls == steps and not (state == model.adapter.vocabulary).any()
    assert torch.equal(state[observations.identity_mask], observations.identity[observations.identity_mask])
    return dict(coordinates=output['sample_atom_coords'], identity=state, identity_logits=logits,
        feedback_calls=calls, native_sampler=True, chemical_schedule='coordinate_noise_cdf')
