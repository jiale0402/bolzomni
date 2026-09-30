"""Shared completion mechanism. Not released for benchmarking until audits pass.

No task selector: behavior depends only on observed fields and chemical state.
The current adapter exposes a canonical-residue vocabulary; output-class extensions
require their own slot/compiler/decoder implementation, not a renamed task switch.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Observations:
    identity: torch.Tensor        # B,N canonical vocabulary IDs, MASK for hidden
    identity_mask: torch.Tensor   # B,N
    coordinates: torch.Tensor     # B,M,3; ZERO outside coordinate_mask
    coordinate_mask: torch.Tensor # B,M; physical observed atoms only

    def validate(self, vocabulary=20):
        assert self.identity_mask.dtype == torch.bool
        assert self.coordinate_mask.dtype == torch.bool
        assert torch.all(self.identity[~self.identity_mask] == vocabulary)
        assert torch.all(self.coordinates[~self.coordinate_mask] == 0)
        assert torch.isfinite(self.coordinates).all()
        assert torch.all((self.identity[self.identity_mask] >= 0) &
                         (self.identity[self.identity_mask] < vocabulary))


class LowRankLinear(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original.requires_grad_(False)
        self.down = nn.Linear(original.in_features, rank, bias=False,
                              device=original.weight.device, dtype=original.weight.dtype)
        self.up = nn.Linear(rank, original.out_features, bias=False,
                            device=original.weight.device, dtype=original.weight.dtype)
        nn.init.zeros_(self.up.weight)
        self.enabled = True

    def forward(self, x):
        y = self.original(x)
        return y + self.up(self.down(x)) if self.enabled else y


class CompletionAdapter(nn.Module):
    def __init__(self, token_dim, pair_dim, hidden_dim, rank=8, vocabulary=20):
        super().__init__()
        self.vocabulary = vocabulary
        self.token_norm = nn.LayerNorm(token_dim)
        # chemical state; identity visibility; coordinate visibility fraction;
        # coordinate sigma and independent chemical corruption level t.
        self.encoder = nn.Sequential(nn.Linear(token_dim+vocabulary+5,128), nn.SiLU())
        self.token_out = nn.Linear(128, token_dim)
        self.pair_left = nn.Linear(128,rank)
        self.pair_right = nn.Linear(128,rank)
        self.pair_out = nn.Linear(rank,pair_dim,bias=False)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_context = nn.Linear(pair_dim,rank,bias=False)
        self.identity_readout = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim,vocabulary))
        nn.init.zeros_(self.token_out.weight); nn.init.zeros_(self.token_out.bias)
        nn.init.zeros_(self.pair_out.weight)

    def forward(self, s, z, state, observations, atom_to_token, sigma, chemical_t):
        observed_atoms = observations.coordinate_mask.float()
        count = atom_to_token.float().sum(1).clamp_min(1)
        coordinate_visibility = torch.bmm(atom_to_token.transpose(1,2).float(),
                                          observed_atoms[...,None]).squeeze(-1)/count
        def broadcast(value):
            return torch.ones_like(coordinate_visibility)[...,None]*value.reshape(-1,1,1)
        h = self.encoder(torch.cat((self.token_norm(s), F.one_hot(state,self.vocabulary+1).float(),
            observations.identity_mask.float()[...,None], coordinate_visibility[...,None],
            broadcast(sigma.clamp_min(1e-8).log()), broadcast(chemical_t)), -1))
        # Directed context is retained: token-pair representations need not be symmetric.
        pair = self.pair_left(h)[:,:,None,:]*self.pair_right(h)[:,None,:,:]
        pair = pair + self.pair_context(self.pair_norm(z))
        return self.token_out(h), self.pair_out(pair)


class OmniCompletion(nn.Module):
    def __init__(self, base, token_dim, pair_dim, hidden_dim, rank=8):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.adapter = CompletionAdapter(token_dim,pair_dim,hidden_dim,rank).to(next(base.parameters()).device)
        self.lora_names = []
        for i, layer in enumerate(base.structure_module.score_model.token_transformer.layers):
            for name in ('proj_q','proj_v'):
                original = getattr(layer.pair_bias_attn,name)
                setattr(layer.pair_bias_attn,name,LowRankLinear(original,rank))
                self.lora_names.append(f'base.structure_module.score_model.token_transformer.layers.{i}.pair_bias_attn.{name}')
        self.live = None
        self._hook = base.structure_module.score_model.a_norm.register_forward_hook(self._capture)

    def _capture(self,module,args,value):
        self.live = value

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @contextmanager
    def native(self):
        modules = [m for m in self.base.modules() if isinstance(m,LowRankLinear)]
        previous = [m.enabled for m in modules]
        try:
            for m in modules: m.enabled = False
            yield
        finally:
            for m,enabled in zip(modules,previous): m.enabled = enabled

    def trainable_state(self):
        names = {n for n,p in self.named_parameters() if p.requires_grad}
        return {n:p.detach().cpu() for n,p in self.state_dict().items() if n in names}

    def denoise(self, cache, noisy, sigma, state, observations, chemical_t, native=False):
        kwargs = dict(cache['conditioning'])
        sigma = torch.as_tensor(sigma,device=noisy.device,dtype=noisy.dtype).reshape(-1)
        if not native:
            ds,dz = self.adapter(kwargs['s_trunk'],cache['z_trunk'],state,observations,
                                kwargs['feats']['atom_to_token'],sigma,chemical_t)
            s = kwargs['s_trunk']+ds
            # Keep trunk static; refresh the small frozen conditioning bridge with
            # both S+deltaS and Z+deltaZ, so pair feedback reaches every native path.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                q,c,keys,enc,dec,token = self.base.diffusion_conditioning(
                    s_trunk=s.to(cache['bridge_s'].dtype),z_trunk=(cache['z_trunk']+dz).to(cache['z_trunk'].dtype),
                    relative_position_encoding=cache['relative_position_encoding'],feats=kwargs['feats'])
            kwargs['s_trunk'] = s
            kwargs['diffusion_conditioning'] = dict(q=q,c=c,to_keys=keys,atom_enc_bias=enc,
                                                   atom_dec_bias=dec,token_trans_bias=token)
        pred,_ = self.base.structure_module.preconditioned_network_forward(
            noisy,sigma,{**kwargs,'multiplicity':1},training=False)
        logits = None if native else self.adapter.identity_readout(self.live)
        return pred,logits


def clamp_coordinates(x, mask, observed):
    return torch.where(mask[...,None],observed,x)


def completion_loss(model, cache, target, target_identity, resolved, observations, seed=None,
                    identity_weight=0.1, replay_weight=1.0, chemical_t_override=None):
    """Paper Eqs 8-12. Replay uses the SAME native input and noisy coordinate state."""
    from boltzgen.model.loss.diffusion import weighted_rigid_align
    observations.validate()
    if seed is not None: torch.manual_seed(seed)
    sm = model.base.structure_module
    sigma = sm.noise_distribution(target.shape[0])
    chemical_t = torch.rand(target.shape[0],device=target.device)
    # Include the two chemical endpoints, independently of coordinate sigma.
    choice = torch.rand_like(chemical_t)
    chemical_t = torch.where(choice<0.25,torch.ones_like(chemical_t),chemical_t)
    chemical_t = torch.where(choice>0.75,torch.zeros_like(chemical_t),chemical_t)
    if chemical_t_override is not None:
        chemical_t = torch.full_like(chemical_t,float(chemical_t_override))
    hidden = (~observations.identity_mask) & (torch.rand_like(target_identity.float())<chemical_t[:,None])
    state = target_identity.clone()
    state[hidden] = model.adapter.vocabulary
    state = torch.where(observations.identity_mask,observations.identity,state)
    noisy = target + sigma[:,None,None]*torch.randn_like(target)
    noisy = clamp_coordinates(noisy,observations.coordinate_mask,observations.coordinates)
    pred,logits = model.denoise(cache,noisy,sigma,state,observations,chemical_t)
    # With absolute anchors, rotating the target to the prediction is not legitimate.
    # Unanchored folding uses inherited rigid-alignment supervision.
    aligned = target
    if not observations.coordinate_mask.any():
        with torch.no_grad():
            aligned = weighted_rigid_align(target.float(),pred.detach().float(),
                                           torch.ones_like(resolved),resolved.float())
    free = resolved.float()*(~observations.coordinate_mask).float()
    xyz = (((pred-aligned).square().sum(-1)*free).sum(-1)/(3*free.sum(-1).clamp_min(1))
           *sm.loss_weight(sigma)).mean()
    ce = F.cross_entropy(logits[hidden],target_identity[hidden]) if hidden.any() else logits.sum()*0
    # Teacher is always unmodified, with no visibility advantage.
    # Replay only on native-compatible prompts, specified by compiler rather than task name.
    if cache.get('native_replay_eligible',False):
        with torch.no_grad(),model.native():
            teacher,_ = model.denoise(cache,noisy,sigma,state,observations,chemical_t,native=True)
        replay = ((pred-teacher).square().sum(-1)*resolved).sum()/(3*resolved.sum().clamp_min(1))
    else:
        replay = pred.sum()*0
    return xyz+identity_weight*ce+replay_weight*replay,dict(coordinate=xyz.detach(),identity=ce.detach(),replay=replay.detach())


@torch.no_grad()
def sample(model, cache, observations, steps=500, seed=1729, step_scale=2.0, noise_scale=0.88, precise_frames=False):
    """Native EDM schedule with progressive chemistry and hard-field projection.

    Augment coordinates AND observed anchors in the same frame. Disable reverse
    alignment only for anchored prompts; return anchors in their supplied frame.
    """
    from boltzgen.model.modules.utils import compute_random_augmentation
    from boltzgen.model.loss.diffusion import weighted_rigid_align
    observations.validate()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    sm = model.base.structure_module
    feats = cache['conditioning']['feats']
    mask = feats['atom_pad_mask'].float()
    if mask.shape[0] != 1: raise ValueError('Initial audited sampler requires batch=1')
    if sm.step_scale_function == 'beta' or sm.noise_scale_function != 'constant':
        raise ValueError('Unqualified schedule; do not silently substitute a native schedule')
    sigmas = sm.sample_schedule_dilated(steps) if sm.sampling_schedule=='dilated' else sm.sample_schedule_af3(steps)
    gammas = torch.where(sigmas>sm.gamma_min,sm.gamma_0,0.0)
    step_scales = step_scale*torch.ones(steps,device=mask.device,dtype=torch.float32)
    noise_scales = noise_scale*torch.ones(steps,device=mask.device,dtype=torch.float32)
    x = sigmas[0]*torch.randn((*mask.shape,3),device=mask.device)
    state = observations.identity.clone()
    chemical_rng = torch.Generator(device=x.device).manual_seed(seed+1777)
    anchored = bool(observations.coordinate_mask.any())
    rotation = torch.eye(3,device=x.device)[None]
    translation = torch.zeros((1,1,3),device=x.device)
    if precise_frames:
        rotation,translation=rotation.double(),translation.double()
    anchors = observations.coordinates.clone()
    x = clamp_coordinates(x,observations.coordinate_mask,anchors)
    final_logits = None
    for k in range(steps):
        ss,ns = step_scales[k],noise_scales[k]
        sigma_prev,sigma_next,gamma = sigmas[k].item(),sigmas[k+1].item(),gammas[k+1].item()
        t_hat = sigma_prev*(1+gamma)
        shift = (x*mask[...,None]).sum(1,keepdim=True)/mask.sum(1,keepdim=True)[...,None]
        x = x-shift
        if anchored:
            anchors = anchors-shift
            translation = translation-shift
        if sm.coordinate_augmentation_inference:
            r,tr = compute_random_augmentation(1,device=x.device,dtype=x.dtype)
            x = (x.double()@r.double()+tr.double()).to(x.dtype) if precise_frames else torch.einsum('bmd,bds->bms',x,r)+tr
            if anchored:
                if precise_frames:
                    anchors=(anchors.double()@r.double()+tr.double()).to(anchors.dtype)
                    rotation=rotation@r.double()
                    translation=translation@r.double()+tr.double()
                else:
                    anchors = anchors@r+tr
                    rotation = rotation@r
                    translation = translation@r+tr
        # Match the pinned native implementation, including its noise-scale convention.
        noise_var = ns**2*(t_hat**2-sigma_prev**2)
        noisy = x+ns*math.sqrt(noise_var)*torch.randn((*mask.shape,3),device=mask.device)
        noisy = clamp_coordinates(noisy,observations.coordinate_mask,anchors)
        chemical_t = x.new_tensor([(steps-k)/steps])
        pred,final_logits = model.denoise(cache,noisy,x.new_tensor([t_hat]),state,observations,chemical_t)
        pred = clamp_coordinates(pred,observations.coordinate_mask,anchors)
        if sm.alignment_reverse_diff and not anchored:
            noisy = weighted_rigid_align(noisy.float(),pred.float(),mask,mask).to(pred)
        denoised_over_sigma = (noisy-pred)/t_hat
        x = noisy+ss*(sigma_next-t_hat)*denoised_over_sigma
        x = clamp_coordinates(x,observations.coordinate_mask,anchors)
        reveal = (state==model.adapter.vocabulary) & (~observations.identity_mask)
        reveal &= torch.rand(state.shape,device=x.device,generator=chemical_rng)<1/(steps-k)
        proposed = torch.multinomial(final_logits.softmax(-1).reshape(-1,model.adapter.vocabulary),
                                    1,generator=chemical_rng).reshape(state.shape)
        state = torch.where(reveal,proposed,state)
        state = torch.where(observations.identity_mask,observations.identity,state)
    if anchored:
        x = ((x.double()-translation)@rotation.transpose(-1,-2)).to(x.dtype) if precise_frames else (x-translation)@rotation.transpose(-1,-2)
        x = clamp_coordinates(x,observations.coordinate_mask,observations.coordinates)
    return dict(coordinates=x,identity=state,identity_logits=final_logits)


@torch.no_grad()
def capture_context(base, prompt, recycling_steps=3):
    """Encode the prompt once; preserve native bridge dtypes for zero-update parity."""
    cache = {}
    class Captured(Exception): pass
    def bridge_hook(module,args,kwargs):
        if args:
            kwargs = dict(zip(('s_trunk','z_trunk','relative_position_encoding','feats'),args)) | kwargs
        cache['bridge_s'] = kwargs['s_trunk'].detach().clone()
        cache['z_trunk'] = kwargs['z_trunk'].detach().clone()
        cache['relative_position_encoding'] = kwargs['relative_position_encoding'].detach().clone()
    def intercept(**kwargs):
        cache['conditioning'] = {k:kwargs[k] for k in ('s_trunk','s_inputs','feats','diffusion_conditioning')}
        raise Captured()
    hook = base.diffusion_conditioning.register_forward_pre_hook(bridge_hook,with_kwargs=True)
    original = base.structure_module.sample
    base.structure_module.sample = intercept
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            try: base(prompt,recycling_steps=recycling_steps)
            except Captured: pass
    finally:
        base.structure_module.sample = original
        hook.remove()
    if 'conditioning' not in cache or 'z_trunk' not in cache:
        raise RuntimeError('Static context capture incomplete')
    return cache


def endpoint_identity_consistency(generated_identity, geometric_identity, editable):
    """Compare the ACTUAL revealed state, not an unrelated final argmax readout."""
    valid = (geometric_identity>=0) & (geometric_identity<20)
    agrees = valid & (generated_identity==geometric_identity)
    return bool(torch.all(agrees[editable])), agrees
