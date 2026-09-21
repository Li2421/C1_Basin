import unittest
import numpy as np
from single_integrator.c1.projection_directional import feasible_projection_direction as derivative
from single_integrator.c1.projection_directional import projection_direction


class ProjectionDirectionalTests(unittest.TestCase):
    def test_halfspace_inward_outward_and_nonadditivity(self):
        A=np.array([[1.,0.]])
        plus,_=derivative([0.,0.],A,[0.],[1.,.2])
        minus,_=derivative([0.,0.],A,[0.],[-1.,-.2])
        np.testing.assert_allclose(plus,[1.,.2],atol=1e-10)
        np.testing.assert_allclose(minus,[0.,-.2],atol=1e-10)
        self.assertGreater(np.linalg.norm(plus+minus),.9)

    def test_moving_constraint_one_sided_difference(self):
        # A(h)=(1,h), b(h)=.3h, y(h)=(0,.1)+h*(-1,.2).
        y=np.array([0.,.1]);d=np.array([-1.,.2]);A=np.array([[1.,0.]])
        v,_=derivative(y,A,[0.],d,dA=[[0.,1.]],db=[.3])
        h=1e-6;a=A[0]+h*np.array([0.,1.]);yh=y+h*d
        projected=yh+a*max(0.,(.3*h-a@yh)/(a@a))
        np.testing.assert_allclose(v,(projected-y)/h,atol=2e-6)

    def test_speed_ball_inward_and_outward(self):
        y=np.array([.5,0.]);A=np.empty((0,2))
        for d in (np.array([1.,.4]),np.array([-1.,.4])):
            v,_=derivative(y,A,[],d)
            h=1e-7;raw=y+h*d;projected=raw/max(1.,np.linalg.norm(raw)/.5)
            np.testing.assert_allclose(v,(projected-y)/h,atol=2e-7)

    def test_corner_and_invalid_exterior(self):
        v,_=derivative([0.,0.],np.eye(2),[0.,0.],[-1.,2.])
        np.testing.assert_allclose(v,[0.,2.],atol=1e-10)
        with self.assertRaises(ValueError):derivative([-1.,0.],[[1.,0.]],[0.],[1.,0.])

    def test_strong_moving_halfspace(self):
        y=np.array([-.2,.1]);p=np.array([0.,.1]);d=np.array([.4,.2])
        A=np.array([[1.,0.]]);da=np.array([[0.,1.]])
        v,_=projection_direction(y,A,[0.],p,d,dA=da,db=[.3])
        h=1e-6;a=A[0]+h*da[0];yh=y+h*d
        actual=yh+a*(.3*h-a@yh)/(a@a)
        np.testing.assert_allclose(v,(actual-p)/h,atol=2e-6)

    def test_strong_ball_curvature(self):
        y=np.array([.8,.3]);p=y*.5/np.linalg.norm(y);d=np.array([-.2,.4])
        v,_=projection_direction(y,np.empty((0,2)),[],p,d)
        h=1e-6;yh=y+h*d;actual=yh*.5/np.linalg.norm(yh)
        np.testing.assert_allclose(v,(actual-p)/h,atol=2e-6)

    def test_strong_and_weak_mixed(self):
        for sign in (-1,1):
            v,info=projection_direction([-.2,0.],np.eye(2),[0.,0.],[0.,0.],[.3,sign*.4])
            np.testing.assert_allclose(v,[0.,max(0.,sign*.4)],atol=1e-7)
            self.assertEqual(info['strong_constraints'],1)

    def test_bad_base_and_degenerate_active_set_rejected(self):
        with self.assertRaises(ValueError):
            projection_direction([-.2,0.],[[1.,0.]],[0.],[.1,0.],[1.,0.])
        with self.assertRaises(ValueError):
            projection_direction([0.,0.],[[1.,0.],[2.,0.]],[0.,0.],[0.,0.],[1.,0.])

    def test_identical_constraint_jets_merge_without_hiding_moving_bounds(self):
        v,info=projection_direction([-.2,.1],[[1.,0.],[1.,0.]],[0.,0.],[0.,.1],[.4,.2],
            dA=[[0.,1.],[0.,1.]],db=[.3,.3])
        np.testing.assert_allclose(v,[.2,.4],atol=1e-8)
        self.assertEqual(info['merged_identical_linear_rows'],1)
        with self.assertRaises(ValueError):
            projection_direction([-.2,.1],[[1.,0.],[1.,0.]],[0.,0.],[0.,.1],[.4,.2],db=[.3,-.3])


if __name__=='__main__':unittest.main()
