"""Separate observed coordinates from supervision without erasing legal geometry.

The pinned TokenDistanceEncoder reads feats['coords'] to derive FOUR backbone
distance channels, not only center_coords. Zeroing this whole field corrupts
legitimate structural conditioning even when the CA distance matrix is retained.
"""
import torch
from boltzgen.data import const


def compile_prompt(base, features, observed_atoms=None, canonical_frame=False):
    f = base.masker(features)
    design = f['design_mask'].bool()
    atom_design = torch.bmm(f['atom_to_token'].float(),design.float()[...,None]).squeeze(-1).bool()
    token_visible = f['token_distance_mask'].bool().any(-1)
    atom_visible = torch.bmm(f['atom_to_token'].float(),token_visible.float()[...,None]).squeeze(-1).bool()
    if observed_atoms is None:
        # Native distance groups expose backbone geometry, not unnamed side chains.
        observed_atoms = atom_visible & f['backbone_mask'].bool() & f['atom_pad_mask'].bool()
    if (observed_atoms & atom_design & ~f['backbone_mask'].bool()).any():
        raise ValueError('Hidden residue side chains cannot silently become observations')
    x = f['coords'].reshape(f['atom_pad_mask'].shape[0],-1,3)
    if canonical_frame and observed_atoms.any():
        # Native featurization centers using ALL target atoms; that translation can
        # depend on hidden side chains. Recover the input PDB frame from an observed
        # CA (center_coords is uncentered), at the declared PDB 0.001-A precision.
        ca = torch.bmm(f['token_to_rep_atom'].float(),x)
        observed_ca = torch.bmm(f['token_to_rep_atom'].float(),observed_atoms.float()[...,None]).squeeze(-1).bool()
        for b in range(x.shape[0]):
            if observed_ca[b].any():
                i = int(observed_ca[b].nonzero()[0])
                shift = f['center_coords'][b,i]-ca[b,i]
                x[b] = torch.round((x[b]+shift)*1000)/1000
            else:
                raise ValueError('Canonical PDB frame requires an observed CA anchor')
    f['coords'] = x.masked_fill(~observed_atoms[...,None],0).reshape_as(f['coords'])
    f['center_coords'] = f['center_coords'].masked_fill(~token_visible[...,None],0)
    f['res_type_clone'] = f['res_type'].clone()
    for key in ('bfactor','plddt','fake_atom_mask'):
        f[key] = torch.zeros_like(f[key])
    f['atom_resolved_mask'] = f['atom_pad_mask'].clone()
    f['ref_chirality'][atom_design] = const.chirality_type_ids['CHI_UNSPECIFIED']
    # Writer bookkeeping depends on the parent atom inventory and is not a neural
    # input. Keep the original map only in the outside-the-model source record.
    f.pop('new_to_old_atomidx',None)
    return f
