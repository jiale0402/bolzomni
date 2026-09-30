import unittest
import numpy as np
import torch
from model import Observations, CompletionAdapter, LowRankLinear, clamp_coordinates, endpoint_identity_consistency
from metrics import ca_lddt, paired_summary


class Components(unittest.TestCase):
    def test_frozen_base_low_rank_gradient(self):
        layer = LowRankLinear(torch.nn.Linear(6,4),2)
        x = torch.randn(3,6)
        self.assertTrue(torch.equal(layer(x),layer.original(x)))
        layer(x).sum().backward()
        self.assertIsNone(layer.original.weight.grad)
        self.assertGreater(layer.up.weight.grad.abs().sum().item(),0)

    def test_pair_and_token_respond_to_chemistry(self):
        torch.manual_seed(4)
        adapter = CompletionAdapter(8,4,12,2)
        obs = Observations(torch.full((1,3),20),torch.zeros(1,3,dtype=torch.bool),
                           torch.zeros(1,3,3),torch.zeros(1,3,dtype=torch.bool))
        obs.validate()
        s,z = torch.randn(1,3,8),torch.randn(1,3,3,4)
        args = (obs,torch.eye(3)[None],torch.ones(1),torch.ones(1))
        ds,dz = adapter(s,z,obs.identity,*args)
        self.assertEqual(ds.abs().sum().item(),0)
        self.assertEqual(dz.abs().sum().item(),0)
        (ds.sum()+dz.sum()).backward()
        self.assertGreater(adapter.pair_out.weight.grad.abs().sum().item(),0)
        with torch.no_grad():
            adapter.token_out.weight.normal_(std=0.1)
            adapter.pair_out.weight.normal_(std=0.1)
        ds,dz = adapter(s,z,obs.identity,*args)
        other = obs.identity.clone(); other[0,1]=7
        ds2,dz2 = adapter(s,z,other,*args)
        self.assertGreater((ds-ds2).abs().max().item(),0)
        self.assertGreater((dz-dz2).abs().max().item(),0)

    def test_observations_and_endpoint_failure(self):
        mask = torch.tensor([[True,False]])
        x,obs = torch.randn(1,2,3),torch.tensor([[[1.,2.,3.],[0.,0.,0.]]])
        projected = clamp_coordinates(x,mask,obs)
        self.assertTrue(torch.equal(projected[mask],obs[mask]))
        self.assertTrue(torch.equal(projected[~mask],x[~mask]))
        self.assertFalse(endpoint_identity_consistency(torch.tensor([1,2]),torch.tensor([1,3]),torch.ones(2,dtype=torch.bool))[0])

    def test_lddt_geometry_and_paired_unit(self):
        true = np.array([[0.,0.,0.],[3.8,0,0],[7.6,0,0]])
        rotated = true[:,[1,0,2]]+12
        self.assertAlmostEqual(ca_lddt(rotated,true),1)
        self.assertLess(ca_lddt(true*3,true),1)
        result = paired_summary([0,0.5,1],[0,0.5,1])
        self.assertFalse(result['positive'])
        self.assertTrue(result['degenerate_interval'])


if __name__=='__main__': unittest.main()
