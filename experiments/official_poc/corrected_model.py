"""Full Omni with native spatial augmentation and native smooth-lDDT supervision.

The original failed experiment stays unchanged. This file is a separate version.
"""
import torch
from torch.nn import functional as F
from full_model import FullOmni, gradient_norm
from model import Observations, clamp_coordinates, sample as original_sample


@torch.no_grad()
def refresh_teacher_bridge(teacher, cache):
    """Trunk tensors are portable; cached mixed-precision bridge outputs need refresh.

    In particular, an H100-created bridge must not be used as the supposedly
    identical teacher against an A100-recomputed student bridge.
    """
    kwargs=cache['conditioning']
    with torch.autocast('cuda',dtype=torch.bfloat16):
        q,c,keys,enc,dec,token=teacher.diffusion_conditioning(
            s_trunk=kwargs['s_trunk'].to(cache['bridge_s'].dtype),
            z_trunk=cache['z_trunk'],relative_position_encoding=cache['relative_position_encoding'],feats=kwargs['feats'])
    kwargs['diffusion_conditioning']=dict(q=q,c=c,to_keys=keys,atom_enc_bias=enc,atom_dec_bias=dec,token_trans_bias=token)


def augmented_target(payload, augment=True):
    from boltzgen.model.modules.utils import compute_random_augmentation
    target = payload['target'].float()
    obs = Observations(**payload['observations'])
    mask = payload['cache']['conditioning']['feats']['atom_pad_mask'].float()
    shift = (target*mask[..., None]).sum(1, keepdim=True)/mask.sum(1, keepdim=True)[..., None]
    target = target-shift
    if augment:
        rotation, translation = compute_random_augmentation(target.shape[0], device=target.device, dtype=target.dtype)
        # TF32 is appropriate for neural matmuls, not rigid-body geometry: even
        # one TF32 3x3 product changed pair distances by 0.01076 A in preflight.
        target = (target.double()@rotation.double()+translation.double()).float()
    # Coordinates, corruption, target and hard observations must share one frame.
    observed = target.masked_fill(~obs.coordinate_mask[..., None], 0)
    return target, Observations(obs.identity, obs.identity_mask, observed, obs.coordinate_mask)


def sample(*args, **kwargs):
    return original_sample(*args, precise_frames=True, **kwargs)


def losses(model, payload, seed=None, chemical_t_override=None, sigma_override=None):
    from boltzgen.model.loss.diffusion import weighted_rigid_align, smooth_lddt_loss
    if seed is not None: torch.manual_seed(seed)
    target, obs = augmented_target(payload)
    labels, resolved = payload['labels'], payload['resolved'].float()
    cache = payload['cache']; obs.validate()
    sm = model.base.structure_module
    sigma = sm.noise_distribution(target.shape[0]) if sigma_override is None else target.new_full((target.shape[0],), sigma_override)
    t = torch.rand(target.shape[0], device=target.device)
    choice = torch.rand_like(t)
    t = torch.where(choice < .25, torch.ones_like(t), t)
    t = torch.where(choice > .75, torch.zeros_like(t), t)
    if chemical_t_override is not None: t = torch.full_like(t, chemical_t_override)
    hidden = (~obs.identity_mask) & (torch.rand_like(labels.float()) < t[:, None])
    state = torch.where(hidden, model.adapter.vocabulary, labels)
    state = torch.where(obs.identity_mask, obs.identity, state)
    noisy = target+sigma[:, None, None]*torch.randn_like(target)
    noisy = clamp_coordinates(noisy, obs.coordinate_mask, obs.coordinates)
    pred, logits = model.denoise(cache, noisy, sigma, state, obs, t)
    aligned = target
    if not obs.coordinate_mask.any():
        with torch.no_grad():
            aligned = weighted_rigid_align(target, pred.detach().float(), torch.ones_like(resolved), resolved)
    free = resolved*(~obs.coordinate_mask).float()
    weight = sm.loss_weight(sigma)
    coordinate = ((((pred.float()-aligned)**2).sum(-1)*free).sum(-1)/(3*free.sum(-1).clamp_min(1))*weight).mean()
    projected = clamp_coordinates(pred.float(), obs.coordinate_mask, obs.coordinates)
    spatial = smooth_lddt_loss(projected, target, torch.zeros_like(resolved), resolved, multiplicity=1)
    identity = F.cross_entropy(logits[hidden].float(), labels[hidden]) if hidden.any() else logits.sum()*0
    native_state, native_t = obs.identity.clone(), torch.ones_like(t)
    with torch.no_grad():
        teacher, _ = model.denoise(cache, noisy, sigma, native_state, obs, native_t, native=True)
    if torch.equal(state, native_state) and torch.equal(t, native_t): student = pred
    else: student, _ = model.denoise(cache, noisy, sigma, native_state, obs, native_t)
    replay_raw = ((((student.float()-teacher.float())**2).sum(-1)*resolved).sum(-1)/(3*resolved.sum(-1).clamp_min(1)))
    replay = (replay_raw*weight).mean()
    return dict(coordinate=coordinate, smooth_lddt=spatial, identity=identity, replay=replay), dict(
        sigma=float(sigma.mean()), chemical_t=float(t.mean()), masked_identities=int(hidden.sum()),
        replay_unweighted=float(replay_raw.detach().mean()))
