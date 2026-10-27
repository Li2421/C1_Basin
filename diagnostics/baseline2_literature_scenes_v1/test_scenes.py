import unittest
from experiment import *
import numpy as np
from states import walls,LiteratureConfig
from flow_scene import LiteratureEnv,LiteratureGeometry
from projection_adapter import empty_geometry_constraints,LiteratureSafetyFilter
from shared_control.hard_projection import HardProjectionConfig,barrier_constraints

class ScenesTest(unittest.TestCase):
    def test_circle_tangency(self):
        w=walls('lit_circle_five');self.assertEqual(w.shape,(128,2,2))
        d=np.abs(np.cross(w[:,1]-w[:,0],-w[:,0]))/np.linalg.norm(w[:,1]-w[:,0],axis=-1)
        np.testing.assert_allclose(d,11,atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(w[0,0])-11,11*(1/np.cos(np.pi/128)-1))
    def test_doorway(self):
        c=LiteratureConfig('lit_doorway',2,.1,.1,.3,404);g=LiteratureGeometry(c)
        self.assertEqual(len(g.walls),8)
        self.assertTrue(g.valid_points(np.array([[1.,.05], [1.,1.3],[1.,-1.2]])).all())
        self.assertFalse(g.valid_points(np.array([[1.,.25]]))[0])
    def test_empty_walls(self):
        s=read(HERE/'PAPER_STATES.json')[0];env=LiteratureEnv(s);ref=np.zeros((3,2));a=LiteratureSafetyFilter()(env.snapshot(),ref).velocity
        np.testing.assert_array_equal(a,ref)
        _,_,info=env.step(a);self.assertEqual(info['termination'],'running');self.assertFalse(info['collision'])
    def test_projection_rows(self):
        s=next(s for s in read(HERE/'PAPER_STATES.json') if s['scene']=='lit_doorway');env=LiteratureEnv(s);cfg=HardProjectionConfig()
        a,l,_=barrier_constraints(env.snapshot(),cfg);b,m,_=empty_geometry_constraints(env.snapshot(),cfg)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(l,m)
    def test_single_robot(self):
        s=dict(scene='lit_narrow_passage',N=1,index=0,positions=[[0,0]],goals=[[1,0]],velocities=[[0,0]])
        c=LiteratureConfig('lit_narrow_passage',1,.15,.2,1,250);env=LiteratureEnv(s,c)
        result=LiteratureSafetyFilter()(env.snapshot(),np.array([[.5,0]]));np.testing.assert_array_equal(result.velocity,[[.5,0]])
        from flow_runtime import correction_for
        from shared_control.basis_families import get_basis_family
        np.testing.assert_array_equal(correction_for(env,result.velocity,np.zeros(3),get_basis_family('orthoflow3')),np.zeros((1,2)))

if __name__=='__main__':unittest.main()
