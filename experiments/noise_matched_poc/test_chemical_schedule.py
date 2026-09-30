"""Scalar algorithm invariants only: no models, tensors, cluster or checkpoints."""
import unittest
from chemical_schedule import chemical_levels, reveal_probabilities, joint_objective


class ChemicalScheduleTests(unittest.TestCase):
    def test_reverse_marginals_and_endpoints(self):
        levels = chemical_levels([4608., 434.706787, 16.688654, 3.155688, .0064],
            sigma_data=16., p_mean=-1.2, p_std=1.5)
        probabilities = reveal_probabilities(levels)
        self.assertEqual(probabilities[-1], 1.)
        remaining = 1.
        for probability, expected in zip(probabilities, levels[1:]):
            self.assertGreaterEqual(probability, 0.)
            self.assertLessEqual(probability, 1.)
            remaining *= 1-probability
            self.assertAlmostEqual(remaining, expected, places=14)
        # Only ~0.13% cumulative reveal at the observed high-noise point,
        # not the prior schedule's ~20%. This is a schedule test, not quality proof.
        self.assertLess(1-levels[1], .002)
        self.assertGreater(1-levels[1], .001)

    def test_repeated_sigmas_do_not_force_extra_reveals(self):
        levels=chemical_levels([10.,10.,1.],sigma_data=16.,p_mean=-1.2,p_std=1.5)
        self.assertEqual(reveal_probabilities(levels)[0], 0.)

    def test_invalid_schedules_fail(self):
        for sigmas in ([], [1.,2.], [1.,0.], [float('nan')]):
            with self.assertRaises(ValueError):
                chemical_levels(sigmas,sigma_data=16.,p_mean=-1.2,p_std=1.5)

    def test_teacher_weight_changes_actual_objective(self):
        parts=dict(coordinate=2.,smooth_lddt=3.,identity=5.,replay=7.)
        self.assertEqual(joint_objective(parts,identity_weight=.2,replay_weight=1.),13.)
        self.assertEqual(joint_objective(parts,identity_weight=.2,replay_weight=.5),9.5)
        parts['identity']=0.
        self.assertEqual(joint_objective(parts,identity_weight=.2,replay_weight=.5),8.5)


if __name__=='__main__':
    unittest.main()
