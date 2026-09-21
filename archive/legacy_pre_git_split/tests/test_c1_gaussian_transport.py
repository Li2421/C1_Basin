import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from single_integrator.c1.risk.gaussian_transport import scalar_parameter_weight as weight

def density(x):return np.exp(-.5*x*x)/np.sqrt(2*np.pi)

class GaussianTransportTests(unittest.TestCase):
    def test_affine_event_probability_gradient(self):
        for theta in (-.7,0.,.4):
            actual,_=quad(lambda x:density(x)*weight([x],[1.],[[0.]],1.,[0.]),-theta,np.inf)
            self.assertAlmostEqual(actual,density(theta),places=10)

    def test_rotation_uses_all_noise_coordinates(self):
        # The event depends solely on coordinate two; conditioning only the
        # first coordinate would miss its boundary. Full transport does not.
        self.assertAlmostEqual(weight([4.,.3],[0.,1.],np.zeros((2,2)),1.,[0.,0.]),.3)

    def test_curved_event_includes_divergence(self):
        # rho=theta+x+0.2*tanh(y); ||grad rho||>=1 globally.
        for theta in (-.4,0.,.4):
            def conditional(y):
                t=np.tanh(y);gy=.2*(1-t*t);hyy=-.4*t*(1-t*t)
                low=-theta-.2*t;s=1+gy*gy
                w0=weight([0.,y],[1.,gy],[[0.,0.],[0.,hyy]],1.,[0.,0.])
                return density(y)*(density(low)/s+ndtr(-low)*w0)
            actual,_=quad(conditional,-10.,10.,epsabs=1e-11)
            exact,_=quad(lambda y:density(y)*density(theta+.2*np.tanh(y)),-10.,10.,epsabs=1e-11)
            self.assertAlmostEqual(actual,exact,places=10)

    def test_mixed_derivative_and_degeneracy(self):
        # rho=theta*x, theta=2: v=x/2, divergence=1/2.
        self.assertAlmostEqual(weight([3.],[2.],[[0.]],3.,[1.]),4.)
        with self.assertRaises(ValueError):weight([1.],[0.],[[0.]],1.,[0.])
        with self.assertRaises(ValueError):weight([0.,0.],[1.,1.],[[0.,1.],[0.,0.]],1.,[0.,0.])

    def test_piecewise_margin_missing_interface_counterexample(self):
        # rho=theta+|z|-1, theta=2: event certain, true probability derivative=0.
        # Branch Hessians omit the interface divergence 2*delta_0.
        branch_estimate,_=quad(lambda z:2*density(z)*weight([z],[1.],[[0.]],1.,[0.]),0.,np.inf)
        self.assertAlmostEqual(branch_estimate,np.sqrt(2/np.pi),places=10)
        self.assertGreater(abs(branch_estimate-0.),.79)

if __name__=='__main__':unittest.main()
