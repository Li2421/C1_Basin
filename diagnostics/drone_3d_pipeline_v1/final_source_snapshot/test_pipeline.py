"""Mechanism tests use actions at frozen states, never unlabeled rollout runs."""
from . import resources  # noqa: F401
import unittest
from pathlib import Path
import jax
import numpy as np
from . import pilot,pipeline
from .terminal import terminal_mask, TerminalServoRuntime


class PipelineChecks(unittest.TestCase):
    def test_observation_has_physical_coordinates_only(self):
        s=pilot.initial_state(17)
        obs=pipeline.observation(s["positions"],s["velocities"],s["goals"])
        self.assertEqual(obs.shape,(2,12))
        np.testing.assert_allclose(obs[:,:3],s["positions"],atol=1e-7)

    def test_vertical_flow_column_and_collinear_degeneracy(self):
        p=np.array([[-.26,0.,.4],[.26,0.,.4]]);g=p[::-1].copy()
        b=pipeline.basis(p,g,np.array([[0.,0.,.1],[0.,0.,0.]]),.5)
        self.assertGreater(b[0,2,1],.3)
        zero=pipeline.basis(p,g,np.zeros((2,3)),.5)
        self.assertTrue(np.all(zero[:,1:,:]==0))

    def test_real_checkpoint_tt_zero_and_ff_certificates(self):
        checkpoint=Path("diagnostics/drone_3d_pipeline_v1/flow/best.pkl")
        if not checkpoint.exists():self.skipTest("run cached-data training first")
        runtime=pipeline.FlowRuntime(checkpoint); s=pilot.initial_state(17)
        p=np.array(s["positions"]);v=np.array(s["velocities"]);g=np.array(s["goals"]);key=jax.random.PRNGKey(7)
        tt,d=runtime.action(p,v,g,key,[0,0,0],"TT",pilot.Config())
        np.testing.assert_allclose(tt,d["safe"],atol=1e-7)
        ff,_=runtime.action(p,v,g,key,[.1,.5,0],"FF",pilot.Config())
        a,b=pilot.constraints(p,pilot.Config())
        self.assertGreaterEqual((a@ff.ravel()-b).min(),-1e-8)
        self.assertLessEqual(np.linalg.norm(ff,axis=1).max(),.5+1e-8)

    def test_terminal_servo_scope_and_safe_goal_convergence(self):
        s=pilot.initial_state(17);p=np.array(s["positions"]);g=np.array(s["goals"])
        self.assertFalse(terminal_mask(p,g,pilot.Config()).any())
        checkpoint=Path("diagnostics/drone_3d_pipeline_v1/flow/best.pkl")
        if not checkpoint.exists():self.skipTest("run cached-data training first")
        runtime=TerminalServoRuntime(checkpoint)
        p=g+np.array([[.1,.01,.02],[-.1,-.01,-.02]])
        self.assertTrue(terminal_mask(p,g,pilot.Config()).all())
        u,_=runtime.action(p,np.zeros((2,3)),g,jax.random.PRNGKey(0),[.5,.5,0],"TT",pilot.Config())
        self.assertTrue((np.sum(u*(g-p),axis=1)>0).all())
        self.assertGreater(pilot.swept_clearances(p,p+.1*u,pilot.Config()).min(),0)
        with self.assertRaisesRegex(ValueError,"only for TT"):
            runtime.action(p,np.zeros((2,3)),g,jax.random.PRNGKey(0),[0,0,0],"FF",pilot.Config())


if __name__=="__main__":unittest.main()
