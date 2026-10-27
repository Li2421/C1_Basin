import unittest
from experiment import *
import numpy as np
import jax
from controllers import Nominal
from flow_scene import LiteratureEnv
from planning import Grid
from states import LiteratureConfig
import ctypes

class ControllerTests(unittest.TestCase):
    def test_all_types_read_count(self):
        state=read(HERE/'PAPER_STATES.json')[0];env=LiteratureEnv(state);rows=read(HERE/'CONTROLLER_MANIFEST.json');seen=set()
        for row in rows:
            if row['type_code'] in seen:continue
            seen.add(row['type_code']);a=Nominal(row).action(env,jax.random.PRNGKey(0))
            self.assertEqual(a.shape,(3,2));self.assertTrue(np.isfinite(a).all());self.assertLessEqual(np.linalg.norm(a,axis=-1).max(),1+1e-8)
    def test_wall_cost_and_gradient(self):
        rows=read(HERE/'CONTROLLER_MANIFEST.json')
        for scene in ('lit_crowded_random','lit_circle_five','lit_doorway'):
            from flow_scene import LiteratureGeometry
            candidates=[s for s in read(HERE/'PAPER_STATES.json') if s['scene']==scene]
            s=candidates[0]
            c=LiteratureConfig(scene,s['N'],2. if scene=='lit_circle_five' else .1 if scene=='lit_doorway' else .15,.05 if scene=='lit_circle_five' else .1,2. if scene=='lit_circle_five' else .3,2000)
            # This checks controller array interfaces only, not a rollout or
            # the experiment's as-yet-unmeasured circle speed parameter.
            geom=LiteratureGeometry(c)
            s=next(s for s in candidates if geom.valid_points(np.asarray(s['positions'])).all() and geom.valid_points(np.asarray(s['goals'])).all())
            env=LiteratureEnv(s,c)
            for kind in ('gradient_mpc','distributed_cross_entropy','centralized_mppi'):
                row=next(r for r in rows if r['type_code']==kind);a=Nominal(row).action(env,jax.random.PRNGKey(0))
                self.assertEqual(a.shape,(s['N'],2));self.assertTrue(np.isfinite(a).all())
    def test_kernel_empty_geometry_exact(self):
        state=read(HERE/'PAPER_STATES.json')[0];env=LiteratureEnv(state);grid=Grid(env,.15,.3)
        old=ctypes.CDLL(str(ROOT/'diagnostics/crosstype6_shared_correction_controller_information_v1/astar_kernel.so'))
        P=ctypes.POINTER(ctypes.c_double);I=ctypes.POINTER(ctypes.c_int)
        old.astar.argtypes=[ctypes.c_double]*4+[ctypes.c_int,P,P,ctypes.c_int,I,P,P];old.astar.restype=ctypes.c_int
        rng=np.random.default_rng(2026102005)
        for _ in range(16):
            start=np.ascontiguousarray(rng.uniform(-1,1,2));goal=np.ascontiguousarray(rng.uniform(-1,1,2));out=np.zeros((702,2));lens=np.zeros(0,np.int32);reserved=np.zeros((0,2))
            length=old.astar(.15,0.,float('inf'),.3,700,start.ctypes.data_as(P),goal.ctypes.data_as(P),0,lens.ctypes.data_as(I),reserved.ctypes.data_as(P),out.ctypes.data_as(P))
            grid.reserved=[];result=grid.search(start,goal);np.testing.assert_array_equal(result,out[:length])

if __name__=='__main__':unittest.main()
