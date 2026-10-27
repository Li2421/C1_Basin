from experiment import *
from paper_secondary import cooptimization,intersection_inputs
import numpy as np
import unittest

class Equations(unittest.TestCase):
    def test_ccs_and_pcca_rows(self):
        angles=np.arange(5)*2*np.pi/5;p=4*np.c_[np.cos(angles),np.sin(angles)];v=-.3*p
        u0=-.7*p;estimate=np.random.default_rng(17).normal(0,.03,(5,5,2));estimate[np.arange(5),np.arange(5)]=0
        for method in ('circle_ccs','circle_pcca'):
            u,virtual,inf,_=cooptimization(p,v,u0,method,estimate)
            self.assertFalse(inf)
            for h in range(5):
                np.testing.assert_allclose(u[h],virtual[h,h]+(u0[h] if method=='circle_ccs' else 0),rtol=0,atol=0)
                for i in range(5):
                    for j in range(i+1,5):
                        d=p[i]-p[j];dv=v[i]-v[j];a=2*(dv@dv)+10*(d@dv)+6*(d@d-16);b=2*d
                        if method=='circle_ccs':extra=2*(u0[h] if i==h else -u0[h] if j==h else np.zeros(2))
                        else:extra=estimate[h,i]-estimate[h,j]
                        F=a+b@(virtual[h,i]-virtual[h,j]+extra)
                        self.assertGreaterEqual(F,-2e-6)

    def test_official_intersection_assignments(self):
        p,g,obs=intersection_inputs();np.testing.assert_array_equal(p,[[0,-2,0],[-2,0,0]]);np.testing.assert_array_equal(g,[[0,1,0],[1,0,0]])
        self.assertIn((-.3,-.6,.1),obs);self.assertEqual(sum(x==(-.3,-.3,.1) for x in obs),2)
        self.assertTrue(all(x[2]==.1 for x in obs))

if __name__=='__main__':unittest.main()
