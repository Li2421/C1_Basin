from experiment import *
from failure_analysis import cone_escape
import numpy as np
import unittest

class ConeGeometry(unittest.TestCase):
    def test_rank_deficient_zero_infimum_is_unattained(self):
        result=cone_escape([-1.,0.],np.array([[1.,0.]]));self.assertEqual(result['infimum'],0.);self.assertFalse(result['attained']);np.testing.assert_allclose(np.array(result['direction'])@np.array([1.,0.]),0.,atol=1e-15)
    def test_full_rank_quadrant_distance(self):
        result=cone_escape([-2.,-3.],np.eye(2));self.assertAlmostEqual(result['infimum'],2.);np.testing.assert_allclose(result['direction'],[1.,0.],atol=1e-15)
    def test_already_outside_and_whole_space(self):
        result=cone_escape([1.,-1.],np.eye(2));self.assertEqual(result['infimum'],0.);self.assertTrue(result['attained']);self.assertIsNone(result['direction'])
        result=cone_escape([1.,-1.],np.array([[1,0],[-1,0],[0,1],[0,-1]]));self.assertIsNone(result['infimum']);self.assertEqual(result['reason'],'cone_is_full_space')

if __name__=='__main__':unittest.main()
