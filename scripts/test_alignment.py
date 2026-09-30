"""Check the evaluation alignment fits only the unedited scaffold."""
import unittest
import numpy as np
from evaluate_panel import align, rmsd


class AlignmentTest(unittest.TestCase):
    def test_rigid_transform_and_local_error(self):
        source = np.random.default_rng(42).normal(size=(20, 3))
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        target = source @ rotation + np.array([3, -2, 5])
        scaffold = np.ones(20, bool)
        scaffold[8:12] = False
        transformed = align(source, target, scaffold)
        self.assertLess(rmsd(transformed, target, scaffold), 1e-10)
        changed = target.copy()
        changed[~scaffold] += np.array([4, 0, 0])
        # Hidden-region displacement must not change the fitted scaffold frame.
        aligned = align(source, changed, scaffold)
        np.testing.assert_allclose(aligned, transformed, atol=1e-10)
        self.assertAlmostEqual(rmsd(aligned, changed, ~scaffold), 4.0)


if __name__ == '__main__':
    unittest.main()
