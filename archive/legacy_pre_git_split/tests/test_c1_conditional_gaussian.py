import math
import unittest
import numpy as np
from scipy.special import ndtr
from single_integrator.c1.risk.conditional_gaussian import interval_probability,implicit_boundary_derivative


class ConditionalGaussianTests(unittest.TestCase):
    def test_halfline_event_probability_and_boundary_gradient(self):
        theta=.3
        p,g=interval_probability([[-theta,np.inf]],[[[-1.],[0.]]])
        self.assertAlmostEqual(p,ndtr(theta),places=14)
        self.assertAlmostEqual(g[0],math.exp(-theta**2/2)/math.sqrt(2*math.pi),places=14)
        p_after,_=interval_probability([[-(theta-.01*g[0]),np.inf]])
        self.assertLess(p_after,p)

    def test_two_boundaries_match_finite_difference(self):
        theta=.4;width=.7
        p,g=interval_probability([[theta-width,theta+width]],[[[1.],[1.]]])
        h=1e-6
        plus,_=interval_probability([[theta+h-width,theta+h+width]])
        minus,_=interval_probability([[theta-h-width,theta-h+width]])
        self.assertAlmostEqual(g[0],(plus-minus)/(2*h),places=9)
        self.assertGreater(p,0.)

    def test_multiple_intervals_without_monotonic_event_assumption(self):
        p,g=interval_probability([[-np.inf,-1.],[1.,np.inf]],[[[0.],[-1.]],[[1.],[0.]]])
        self.assertAlmostEqual(p,2*ndtr(-1.),places=14)
        self.assertAlmostEqual(g[0],-2*math.exp(-.5)/math.sqrt(2*math.pi),places=14)

    def test_tail_mass_and_empty_event(self):
        p,_=interval_probability([[9.,np.inf]])
        self.assertGreater(p,0.)
        self.assertAlmostEqual(p/ndtr(-9.),1.,places=14)
        p,g=interval_probability(np.empty((0,2)),np.empty((0,2,3)))
        self.assertEqual(p,0.);np.testing.assert_array_equal(g,np.zeros(3))

    def test_implicit_root_and_rejected_bad_boundaries(self):
        np.testing.assert_allclose(implicit_boundary_derivative([1.,2.],-2.),[.5,1.])
        with self.assertRaises(ValueError):implicit_boundary_derivative([1.],0.)
        with self.assertRaises(ValueError):interval_probability([[0.,2.],[1.,3.]])
        with self.assertRaises(ValueError):interval_probability([[0.,np.inf]],[[[1.],[1.]]])


if __name__=='__main__':unittest.main()
