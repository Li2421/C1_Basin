"""A failed solver flag is acceptable only with the original certificates."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from single_integrator.cbf import CBFConfig, CBFSolverError, project_velocity


class SolverStatusTests(unittest.TestCase):
    def solve(self, candidate):
        result = SimpleNamespace(x=np.array(candidate), success=False,
                                 message='Inequality constraints incompatible')
        with patch('single_integrator.cbf.minimize', return_value=result):
            return project_velocity([-.1, .1, 0, 0], np.array([[1., 0, 0, 0]]),
                                    np.zeros(1), .5, CBFConfig())

    def test_certified_optimum_is_accepted(self):
        value, status = self.solve([0, .1, 0, 0])
        np.testing.assert_array_equal(value.ravel(), [0, .1, 0, 0])
        self.assertEqual(status, 'solved_kkt_certified')

    def test_feasible_but_nonoptimal_is_rejected(self):
        with self.assertRaises(CBFSolverError):
            self.solve([0, 0, 0, 0])

    def test_infeasible_is_rejected(self):
        with self.assertRaises(CBFSolverError):
            self.solve([-.01, .1, 0, 0])

    def test_nonfinite_is_rejected(self):
        with self.assertRaises(CBFSolverError):
            self.solve([float('nan'), .1, 0, 0])
