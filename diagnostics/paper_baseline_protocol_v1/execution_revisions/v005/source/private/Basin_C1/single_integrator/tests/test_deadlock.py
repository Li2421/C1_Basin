import unittest
import numpy as np
from single_integrator.environment import Config, GiveWayEnv


class WindowDeadlockTests(unittest.TestCase):
    def step_many(self, env, action, n):
        for _ in range(n):
            _, _, done, info = env.step(action)
        return done, info

    def test_waiting_agent_is_not_system_deadlock(self):
        env = GiveWayEnv()
        done, info = self.step_many(env, [[0., 0.], [-.03, 0.]], 150)
        self.assertFalse(done)
        self.assertFalse(info['candidate_deadlock'])
        self.assertEqual(info['stuck_timer'], 0.)

    def test_negative_progress_at_low_speed_prevents_candidate(self):
        env = GiveWayEnv()
        done, info = self.step_many(env, [[-.02, 0.], [0., 0.]], 140)
        self.assertFalse(done)
        self.assertLess(info['window_progress'][0], -.01)
        self.assertFalse(info['candidate_deadlock'])

    def test_temporary_pause_and_speed_reset(self):
        env = GiveWayEnv()
        _, info = self.step_many(env, np.zeros((2, 2)), 100)
        self.assertAlmostEqual(info['stuck_timer'], 3.)
        _, _, done, info = env.step([[.03, 0.], [0., 0.]])
        self.assertFalse(done)
        self.assertEqual(info['stuck_timer'], 0.)
        _, info = self.step_many(env, np.zeros((2, 2)), 40)
        self.assertAlmostEqual(info['stuck_timer'], 1.95)
        self.assertFalse(info['deadlock'])

    def test_progress_resets_timer_even_with_low_current_speed(self):
        env = GiveWayEnv(Config(progress_epsilon=.0005))
        self.step_many(env, np.zeros((2, 2)), 100)
        _, _, _, info = env.step([[.02, 0.], [0., 0.]])
        self.assertLess(info['max_speed'], .025)
        self.assertFalse(info['candidate_deadlock'])
        self.assertEqual(info['stuck_timer'], 0.)

    def test_one_finished_agent_does_not_mask_other_stuck_agent(self):
        env = GiveWayEnv(); env.reset([[2.29, 0.], [-1., 0.]])
        done, info = self.step_many(env, np.zeros((2, 2)), 140)
        self.assertTrue(done)
        self.assertEqual(info['termination'], 'deadlock')
        self.assertEqual(env.summary()['deadlock_trigger_timestep'], 140)

    def test_success_suppresses_candidate_and_ever_deadlock_is_latched(self):
        env = GiveWayEnv(Config(terminate_on_success=False))
        env.reset([[2.29, 0.], [-2.29, 0.]])
        _, info = self.step_many(env, np.zeros((2, 2)), 150)
        self.assertFalse(info['candidate_deadlock'])
        env = GiveWayEnv(Config(terminate_on_deadlock=False))
        self.step_many(env, np.zeros((2, 2)), 140)
        _, _, _, info = env.step([[.03, 0.], [0., 0.]])
        self.assertFalse(info['deadlock'])
        self.assertTrue(env.summary()['deadlock'])
        self.assertEqual(env.summary()['deadlock_trigger_timestep'], 140)
        env.reset()
        self.assertFalse(env.summary()['deadlock'])
        self.assertEqual(len(env.distance_history), 1)

    def test_collision_stops_at_first_event(self):
        for initial, action, kind in [
            ([[-1., .025], [1., 0.]], [[0., .4], [0., 0.]], 'wall_collision'),
            ([[-.17, 0.], [.17, 0.]], [[.4, 0.], [-.4, 0.]], 'agent_collision'),
        ]:
            env = GiveWayEnv(); env.reset(initial)
            _, _, done, info = env.step(action)
            self.assertTrue(done)
            self.assertTrue(info[kind])
            self.assertEqual(info['termination'], 'collision')
            with self.assertRaises(RuntimeError): env.step(action)

    def test_fractional_window_and_strict_threshold(self):
        env = GiveWayEnv(Config(progress_window_seconds=.075))
        _, _, _, info = env.step([[.025, 0.], [0., 0.]])
        self.assertFalse(info['window_ready'])
        _, _, _, info = env.step([[.025, 0.], [0., 0.]])
        np.testing.assert_allclose(info['window_progress'], [.075*.025, 0.], atol=1e-12)
        self.assertFalse(info['candidate_deadlock'])


if __name__ == '__main__': unittest.main()
