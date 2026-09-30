"""Real checkpoint tests, on a validation example only; no optimizer run."""
import json
from pathlib import Path
import torch
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals
from boltzgen.task.predict.data_from_yaml import Dataset, PredictionDataset, collate
from poc_model import ROOT, ASSETS, load_model, move, safe_prompt, labels
from model import OmniCompletion, Observations, capture_context, completion_loss, sample
from prompt import compile_prompt


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(1729)
    row = json.loads((ROOT/'manifests/data.json').read_text())['validation'][0]
    base = load_model()
    ds = PredictionDataset(Dataset(str(Path(row['directory'])/'fixed.yaml'),
        Tokenizer(atomize_modified_residues=False),Featurizer()),load_canonicals(str(ASSETS/'mols.zip')),
        str(ASSETS/'mols.zip'),atom14=True,design=True,disulfide_on=False)
    f = move(collate([ds[0]]),'cuda')
    target = f['coords'].reshape(1,-1,3).clone()
    lab = labels(f)
    prompt = compile_prompt(base,f,canonical_frame=True)
    cache = capture_context(base,prompt)
    cache['native_replay_eligible'] = True
    kwargs = cache['conditioning']
    obs = Observations(lab.clone(),~f['design_mask'].bool(),torch.zeros_like(target),
                       torch.zeros_like(f['atom_pad_mask'],dtype=torch.bool))
    obs.identity[~obs.identity_mask] = 20
    obs.validate()
    model = OmniCompletion(base,kwargs['s_trunk'].shape[-1],cache['z_trunk'].shape[-1],
                           base.structure_module.score_model.a_norm.normalized_shape[0]).cuda().eval()
    sigma = torch.tensor([4.],device='cuda')
    noisy = target+sigma[:,None,None]*torch.randn_like(target)
    with torch.no_grad(),model.native():
        native,_ = model.denoise(cache,noisy,sigma,obs.identity,obs,torch.ones_like(sigma),native=True)
    with torch.no_grad():
        zero,_ = model.denoise(cache,noisy,sigma,obs.identity,obs,torch.ones_like(sigma))
    diff = float((zero-native).abs().max())
    print('ZERO_DENOISER_MAX_ERROR',diff,flush=True)
    assert diff<1e-5, 'New zero-update bridge must preserve native output'
    model.zero_grad(set_to_none=True)
    value,parts = completion_loss(model,cache,target,lab,f['atom_resolved_mask'].float(),obs,
                                 seed=1729,chemical_t_override=1.0)
    value.backward()
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    active = {n for n,p in trainable.items() if p.grad is not None and p.grad.abs().sum()>0}
    assert any('pair_out' in n for n in active)
    assert any('token_out' in n for n in active)
    assert any('.up.weight' in n for n in active)
    assert any('identity_readout' in n for n in active)
    assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
    print('GRADIENTS_PASS',len(active),sum(p.numel() for p in trainable.values()),flush=True)
    model.zero_grad(set_to_none=True)
    with torch.no_grad(),model.native():
        torch.manual_seed(99); torch.cuda.manual_seed_all(99)
        native_sample = base.structure_module.sample(**kwargs,atom_mask=f['atom_pad_mask'].float(),
                    multiplicity=1,num_sampling_steps=8,step_scale=2.0,noise_scale=0.88)['sample_atom_coords']
    generated = sample(model,cache,obs,steps=8,seed=99)
    sample_error = float((generated['coordinates']-native_sample).abs().max())
    print('ZERO_SAMPLER_MAX_ERROR',sample_error,flush=True)
    assert sample_error<1e-4
    # Hard-frame observation handling: fix all backbone atoms of observed residues.
    atom_known = torch.bmm(f['atom_to_token'].float(),obs.identity_mask.float()[...,None]).squeeze(-1).bool()
    hard = atom_known & f['backbone_mask'].bool() & f['atom_resolved_mask'].bool()
    anchored = Observations(obs.identity,obs.identity_mask,target.masked_fill(~hard[...,None],0),hard)
    result = sample(model,cache,anchored,steps=8,seed=99)
    assert torch.equal(result['coordinates'][hard],target[hard])
    assert torch.equal(result['identity'][obs.identity_mask],lab[obs.identity_mask])
    print('HARD_OBSERVATION_PASS',flush=True)
    # Perturb feedback only after zero-initialization checks, without changing base weights.
    with torch.no_grad():
        model.adapter.token_out.weight.normal_(std=0.01)
        model.adapter.pair_out.weight.normal_(std=0.01)
        state2 = obs.identity.clone(); state2[~obs.identity_mask] = 7
        y1,_ = model.denoise(cache,noisy,sigma,obs.identity,obs,torch.ones_like(sigma))
        y2,_ = model.denoise(cache,noisy,sigma,state2,obs,torch.ones_like(sigma))
    feedback_effect = float((y1-y2).abs().max())
    assert feedback_effect>0, 'Chemical feedback must affect subsequent coordinates'
    print('FEEDBACK_EFFECT',feedback_effect,flush=True)
    output = dict(zero_denoiser_error=diff,zero_sampler_error=sample_error,
                  trainable_parameters=sum(p.numel() for p in trainable.values()),
                  feedback_effect=feedback_effect,hard_observations=True,gradients=True)
    path = ROOT/'experiments/omni_joint/results/integration_audit.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2))
    print('INTEGRATION_PASS',flush=True)


if __name__=='__main__': main()
