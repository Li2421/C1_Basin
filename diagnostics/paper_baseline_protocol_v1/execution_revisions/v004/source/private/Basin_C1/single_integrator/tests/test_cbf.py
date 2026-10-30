import unittest
import numpy as np
from single_integrator.cbf import CBFConfig, CBFSafetyFilter, CBFSolverError, barrier_constraints, project_velocity
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import rollout
from single_integrator.evaluate_cbf import transition_matrix
from single_integrator.outcomes import first_event, first_event_aggregate


class HeadOn:
    def sample_actions(self, observations, seed):
        return np.array([[[.5,0],[-.5,0]]])


class CBFTests(unittest.TestCase):
    def setUp(self):
        self.cfg=CBFConfig()
        self.env=GiveWayEnv(Config())
        self.env.reset([[-.2,0],[.2,0]])

    def test_pair_analytic_projection(self):
        A,b,g=barrier_constraints(self.env.snapshot(),self.cfg)
        u,status=project_velocity([[.5,0],[-.5,0]],A,b,.5,self.cfg)
        expected=g['pairwise_h']/1.6
        np.testing.assert_allclose(u,[[expected,0],[-expected,0]],atol=1e-9)
        self.assertGreaterEqual((A@u.ravel()-b).min(),-1e-9)

    def test_wall_and_opening(self):
        self.env.reset([[-1,.02],[1,0]])
        u=CBFSafetyFilter()(self.env.snapshot(),np.array([[0,.2],[0,0]])).velocity
        self.assertAlmostEqual(u[0,1],.18-.16-4/600-.005-.0001,places=9)
        self.env.reset([[0,.3],[1,0]])
        nominal=np.array([[0,.05],[0,0]])
        np.testing.assert_array_equal(CBFSafetyFilter()(self.env.snapshot(),nominal).velocity,nominal)

    def test_infeasible_has_no_fallback(self):
        with self.assertRaises(CBFSolverError):
            project_velocity(np.zeros((2,2)),np.array([[1,0,0,0],[-1,0,0,0]]),np.ones(2),.5,self.cfg)

    def test_speed_ball_and_optimal_objective(self):
        from scipy.optimize import minimize
        env=GiveWayEnv(Config(wall_layout_scale=5))
        env.reset([[-.18,0],[.18,0]])
        A,b,_=barrier_constraints(env.snapshot(),self.cfg)
        target=np.array([.5,0,.4,.3])
        u,_=project_velocity(target,A,b,.5,self.cfg)
        reference=minimize(lambda x:.5*np.sum((x-target)**2),np.zeros(4),method='SLSQP',
                           constraints=[{'type':'ineq','fun':lambda x:A@x-b},
                                        {'type':'ineq','fun':lambda x:.25-np.sum(x.reshape(2,2)**2,axis=1)}],
                           options={'ftol':1e-12,'maxiter':500})
        self.assertTrue(reference.success)
        self.assertLessEqual(np.linalg.norm(u,axis=1).max(),.5+1e-10)
        self.assertAlmostEqual(.5*np.sum((u.ravel()-target)**2),reference.fun,places=8)

    def test_collision_filter_and_matrix(self):
        config=Config(max_steps=300)
        initial=[[-.3,0],[.3,0]]
        a,aa=rollout(HeadOn(),initial,config,42,0,cbf_config=self.cfg)
        b,bb=rollout(HeadOn(),initial,config,42,0,CBFSafetyFilter(),cbf_config=self.cfg)
        self.assertEqual(a['outcome'],'agent_collision')
        self.assertEqual(b['outcome'],'safe_deadlock')
        self.assertFalse(bb['wall_collision'].any() or bb['agent_collision'].any())
        self.assertGreaterEqual(b['min_pairwise_h'],-1e-9)
        self.assertEqual(sum(transition_matrix([a],[b])['counts'][2]),1)
        agg=first_event_aggregate([a,b])
        self.assertAlmostEqual(sum(v for k,v in agg.items() if k.endswith('_rate') and k!='cbf_intervention_rate'),1.)
        self.assertIsNone(agg['mean_completion_time'])

    def test_simultaneous_and_running(self):
        env=GiveWayEnv(Config(dt=1))
        env.reset([[-.17,0],[.17,0]])
        _,_,done,info=env.step(np.array([[.45,-.2],[-.45,-.2]]))
        self.assertTrue(done and info['wall_collision'] and info['agent_collision'])
        self.assertEqual(first_event(info),'agent_collision')
        info.update(agent_collision=False,wall_collision=False,task_success=False,deadlock=False,termination=None)
        with self.assertRaises(ValueError):first_event(info)


if __name__=='__main__':unittest.main()
