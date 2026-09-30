"""Full diffusion adaptation; frozen trunk and independent frozen native teacher."""
import torch
from torch import nn
from torch.nn import functional as F
from model import OmniCompletion, CompletionAdapter, Observations, clamp_coordinates


class FullOmni(OmniCompletion):
    def __init__(self, base, teacher, cache):
        # Deliberately do not call the LoRA-inserting parent constructor.
        nn.Module.__init__(self)
        self.base = base.eval().requires_grad_(False)
        self.base.structure_module.requires_grad_(True)
        self.base.diffusion_conditioning.requires_grad_(True)
        object.__setattr__(self, 'teacher', teacher.eval().requires_grad_(False))
        self.adapter = CompletionAdapter(
            cache['conditioning']['s_trunk'].shape[-1], cache['z_trunk'].shape[-1],
            base.structure_module.score_model.a_norm.normalized_shape[0], rank=8
        ).to(next(base.parameters()).device)
        self.lora_names = []
        self.live = None
        self._hook = base.structure_module.score_model.a_norm.register_forward_hook(self._capture)
        assert all(type(m).__name__ != 'LowRankLinear' for m in self.modules())
        allowed = ('base.structure_module.', 'base.diffusion_conditioning.', 'adapter.')
        assert all(n.startswith(allowed) for n, p in self.named_parameters() if p.requires_grad)
        assert not any(p.requires_grad for p in teacher.parameters())

    def denoise(self, cache, noisy, sigma, state, observations, chemical_t, native=False):
        # Match teacher/student precision during training. Evaluation retains FP32
        # score calls and the inherited BF16 bridge, as in the original paired assay.
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.training):
            if native:
                kwargs = dict(cache['conditioning'], multiplicity=1)
                pred, _ = self.teacher.structure_module.preconditioned_network_forward(
                    noisy, torch.as_tensor(sigma, device=noisy.device).reshape(-1), kwargs,
                    training=False)
                return pred, None
            return super().denoise(cache, noisy, sigma, state, observations, chemical_t)


def losses(model, payload, seed=None, chemical_t_override=None):
    """Return differentiable, separately measurable objective components.

    Replay uses only observed identities, independently of target-revealed training
    states. Both coordinate MSEs use native sigma weighting and per-coordinate means.
    No extra identity-consistency objective or sampler modification is introduced.
    """
    from boltzgen.model.loss.diffusion import weighted_rigid_align
    if seed is not None:
        torch.manual_seed(seed)
    cache = payload['cache']
    target, labels, resolved = payload['target'], payload['labels'], payload['resolved']
    obs = Observations(**payload['observations'])
    obs.validate()
    assert payload['mode'] in ('folding', 'design')
    sm = model.base.structure_module
    sigma = sm.noise_distribution(target.shape[0])
    t = torch.rand(target.shape[0], device=target.device)
    choice = torch.rand_like(t)
    t = torch.where(choice < .25, torch.ones_like(t), t)
    t = torch.where(choice > .75, torch.zeros_like(t), t)
    if chemical_t_override is not None:
        t = torch.full_like(t, chemical_t_override)
    hidden = (~obs.identity_mask) & (torch.rand_like(labels.float()) < t[:, None])
    state = torch.where(hidden, model.adapter.vocabulary, labels)
    state = torch.where(obs.identity_mask, obs.identity, state)
    noisy = target + sigma[:, None, None] * torch.randn_like(target)
    noisy = clamp_coordinates(noisy, obs.coordinate_mask, obs.coordinates)
    pred, logits = model.denoise(cache, noisy, sigma, state, obs, t)
    aligned = target
    if not obs.coordinate_mask.any():
        with torch.no_grad():
            aligned = weighted_rigid_align(target.float(), pred.detach().float(),
                                           torch.ones_like(resolved), resolved.float())
    free = resolved.float() * (~obs.coordinate_mask).float()
    sigma_weight = sm.loss_weight(sigma)
    coordinate = ((((pred.float() - aligned.float())**2).sum(-1)*free).sum(-1)
                  / (3*free.sum(-1).clamp_min(1)) * sigma_weight).mean()
    identity = F.cross_entropy(logits[hidden].float(), labels[hidden]) if hidden.any() else logits.sum()*0
    # This does not consult the old, folding-only cache eligibility flag.
    native_state = obs.identity.clone()
    assert torch.all(native_state[~obs.identity_mask] == model.adapter.vocabulary)
    native_t = target.new_ones(target.shape[0])
    with torch.no_grad():
        teacher, _ = model.denoise(cache, noisy, sigma, native_state, obs, native_t, native=True)
    if torch.equal(state, native_state) and torch.equal(t, native_t):
        native_student = pred
    else:
        native_student, _ = model.denoise(cache, noisy, sigma, native_state, obs, native_t)
    replay_raw = ((((native_student.float()-teacher.float())**2).sum(-1)*resolved).sum(-1)
                  / (3*resolved.sum(-1).clamp_min(1)))
    replay = (replay_raw*sigma_weight).mean()
    return dict(coordinate=coordinate, identity=identity, replay=replay), dict(
        replay_unweighted=float(replay_raw.detach().mean()), sigma=float(sigma.mean()),
        masked_identities=int(hidden.sum()), replay_eligible=True)


def gradient_norm(value, parameters, retain_graph=True):
    grads = torch.autograd.grad(value, parameters, retain_graph=retain_graph, allow_unused=True)
    total = value.new_zeros((), dtype=torch.float32)
    for g in grads:
        if g is not None:
            total += g.detach().float().square().sum()
    return float(total.sqrt())
