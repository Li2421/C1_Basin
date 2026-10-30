import subprocess
import sys
import unittest
import numpy as np
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal, segment_distance
from single_integrator.expert import Expert


class DynamicsTests(unittest.TestCase):
    def test_exact_integrator_and_instant_stop(self):
        env=GiveWayEnv();before=env.positions.copy();u=np.array([[.2,.01],[-.3,-.01]])
        _,_,_,info=env.step(u)
        np.testing.assert_array_equal(env.positions,before+.05*u)
        np.testing.assert_array_equal(env.velocities,u)
        self.assertEqual(info['tracking_error'],0.)
        before=env.positions.copy();env.step(np.zeros((2,2)))
        np.testing.assert_array_equal(env.positions,before)
        np.testing.assert_array_equal(env.velocities,np.zeros((2,2)))

    def test_no_hidden_clipping(self):
        env=GiveWayEnv();p=env.positions.copy()
        with self.assertRaises(ValueError):env.step(np.array([[.51,0.],[0.,0.]]))
        np.testing.assert_array_equal(env.positions,p)
        self.assertEqual(env.step_count,0)
        np.testing.assert_allclose(bounded_nominal([[3.,4.],[0.,0.]],.5),[[.3,.4],[0.,0.]])

    def test_wall_collision_does_not_change_motion(self):
        env=GiveWayEnv();env.reset([[-1.,.025],[1.,0.]])
        p=env.positions.copy();u=np.array([[0.,.4],[0.,0.]])
        _,_,_,info=env.step(u)
        self.assertTrue(info['wall_collision'])
        np.testing.assert_array_equal(env.positions,p+.05*u)
        np.testing.assert_array_equal(env.velocities,u)

    def test_agents_can_overlap_without_hidden_response(self):
        env=GiveWayEnv();env.reset([[-.17,0.],[.17,0.]])
        u=np.array([[.4,0.],[-.4,0.]]);p=env.positions.copy()
        _,_,_,info=env.step(u)
        self.assertTrue(info['agent_collision'])
        np.testing.assert_array_equal(env.positions,p+.05*u)

    def test_swept_detection_catches_between_step_crossing(self):
        env=GiveWayEnv(Config(dt=1.));env.reset([[-.3,0.],[.3,0.]])
        _,_,_,info=env.step([[.5,0.],[-.5,0.]])
        self.assertFalse(info['endpoint_agent_collision'])
        self.assertTrue(info['agent_collision'])
        self.assertAlmostEqual(float(segment_distance(np.array([0.,0.]),np.array([1.,1.]),np.array([0.,1.]),np.array([1.,0.]))),0.)

    def test_deadlock_timestamp_and_termination(self):
        env=GiveWayEnv()
        for i in range(140):
            _,_,done,info=env.step(np.zeros((2,2)))
            self.assertEqual(done,i==139)
        self.assertEqual(env.first_deadlock_step,140)
        self.assertEqual(info['time'],7.)
        with self.assertRaises(RuntimeError):env.step(np.zeros((2,2)))

    def test_wall_scaling_preserves_policy_coordinates(self):
        original=GiveWayEnv();expanded=GiveWayEnv(Config(wall_layout_scale=1.5))
        np.testing.assert_array_equal(expanded.walls,1.5*original.walls)
        np.testing.assert_array_equal(expanded.observation(),original.observation())
        np.testing.assert_array_equal(expanded.goals,original.goals)
        self.assertFalse(expanded.outside(np.array([[3.,0.],[-3.,0.]])).any())
        self.assertTrue(original.outside(np.array([[3.,0.],[-3.,0.]])).all())
        original.reset([[-1.,.025],[1.,0.]])
        expanded.reset([[-1.,.025],[1.,0.]])
        u=[[0.,.4],[0.,0.]]
        self.assertTrue(original.step(u)[2])
        self.assertFalse(expanded.step(u)[2])
        np.testing.assert_array_equal(original.positions,expanded.positions)

    def test_snapshot_is_detached(self):
        env=GiveWayEnv();snapshot=env.snapshot();snapshot['positions'][:]=999
        self.assertLess(env.positions.max(),999)

    def test_paired_expert_completes_without_collisions(self):
        for mode in [0,1]:
            env=GiveWayEnv();expert=Expert(env,mode)
            for _ in range(850):
                _,_,done,info=env.step(expert.action())
                self.assertFalse(info['wall_collision'])
                self.assertFalse(info['agent_collision'])
                if done:break
            self.assertEqual(info['termination'],'success')

    def test_environment_import_does_not_load_pid_or_physics(self):
        subprocess.run([sys.executable,'-c',"import sys; import single_integrator.environment; import single_integrator.expert; assert 'torch' not in sys.modules; assert not any(k.startswith('vmas') for k in sys.modules)"],check=True)


if __name__=='__main__':unittest.main()
