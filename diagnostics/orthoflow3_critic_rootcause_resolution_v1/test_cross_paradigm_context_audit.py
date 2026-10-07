"""Numerical contracts for the diagnostic physical response summary only.

These do not assert that the underlying slot-dependent controllers are rotation
or permutation invariant, or that a pooled summary preserves all information.
"""
import unittest
import numpy as np
from .cross_paradigm_cached_audit import response


class PhysicalResponseContract(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261004)

    def test_common_physical_coordinate_changes(self):
        for count in (2, 4, 7):
            x, g, u = self.rng.normal(size=(3, count, 2))
            value = response(x, g, u)
            perm = self.rng.permutation(count)
            np.testing.assert_allclose(response(x[perm], g[perm], u[perm]), value, atol=1e-12)
            angle = .71
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            shift = np.array([3.2, -.4])
            np.testing.assert_allclose(response(x@rot+shift, g@rot+shift, u@rot), value, atol=1e-12)

    def test_scaling_and_stationary_response(self):
        x, g, u = self.rng.normal(size=(3, 4, 2))
        value = response(x, g, u)
        np.testing.assert_allclose(response(x, g, 2*u), 2*value, atol=1e-12)
        np.testing.assert_array_equal(response(x, g, np.zeros_like(u)), np.zeros(6))

    def test_goal_zero_and_coincident_position_finiteness(self):
        x = np.zeros((4, 2))
        u = self.rng.normal(size=(4, 2))
        assert np.isfinite(response(x, x, u)).all()


if __name__ == '__main__':
    unittest.main()
