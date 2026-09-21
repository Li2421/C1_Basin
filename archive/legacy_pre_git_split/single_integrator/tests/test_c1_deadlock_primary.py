import unittest
import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.deadlock_primary import temporal_upper, trajectory


class DeadlockPrimaryTests(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64', True)

    def test_bound_against_independent_window_enumeration(self):
        rng = np.random.default_rng(1703)
        for length, predicates, hold in [(30, 3, 5), (17, 7, 4), (8, 1, 8)]:
            x = rng.normal(.2, .4, (length, predicates))
            valid = np.ones(length, bool)
            valid[:2] = False if length > hold else True
            exact = max(x[i:i+hold].min() for i in range(length-hold+1) if valid[i:i+hold].all())
            r = temporal_upper(jnp.asarray(x), jnp.asarray(valid), hold_samples=hold)
            self.assertAlmostEqual(float(r['robustness_hard']), exact)
            self.assertGreaterEqual(float(r['robustness_upper']), exact-1e-12)
            self.assertLessEqual(float(r['robustness_upper'])-exact, float(r['approximation_bound'])+1e-12)
            if exact >= 0:
                self.assertGreaterEqual(float(r['J_live']), 1.)

    def test_masked_and_short_episodes_have_zero_finite_gradient(self):
        for n in (3, 20):
            fn = lambda x: temporal_upper(x, jnp.zeros(n, bool), hold_samples=5)['J_live']
            x = jnp.ones((n, 3))
            self.assertEqual(float(fn(x)), 0.)
            np.testing.assert_array_equal(jax.grad(fn)(x), 0.)

    def test_gradient_spreads_over_temporal_window_and_matches_difference(self):
        x = jnp.linspace(.05, .15, 30).reshape(10, 3)
        fn = lambda a: temporal_upper(a, jnp.ones(10, bool), hold_samples=5)['J_live']
        g = jax.grad(fn)(x)
        self.assertTrue(np.all(np.asarray(g) > 0))
        direction = jnp.sin(jnp.arange(30)).reshape(x.shape)
        h = 1e-6
        fd = (fn(x+h*direction)-fn(x-h*direction))/(2*h)
        self.assertAlmostEqual(float(fd), float(jnp.sum(g*direction)), places=7)

    def test_rotated_translated_three_agent_task_has_same_score(self):
        rng = np.random.default_rng(23)
        start = rng.normal(size=(3, 2))
        u = rng.normal(scale=.001, size=(50, 3, 2))
        positions = np.concatenate([start[None], start[None]+np.cumsum(u*.1, axis=0)])
        goals = start+2
        kwargs = dict(dt=.1, max_speed=.7, goal_tolerance=.1, hold_seconds=1.,
            progress_window_seconds=.5, progress_epsilon=.02, speed_epsilon_fraction=.05)
        def f(x, g, v):
            return trajectory(jnp.asarray(x[:-1]), jnp.asarray(x[1:]), jnp.asarray(v),
                              jnp.asarray(g), jnp.ones(50, bool), **kwargs)['J_live']
        rotation = np.array([[0., -1.], [1., 0.]])
        self.assertAlmostEqual(float(f(positions, goals, u)),
            float(f(positions@rotation+7, goals@rotation+7, u@rotation)), places=10)
        self.assertGreaterEqual(float(f(positions, goals, u)), 1.)

    def test_first_event_matches_environment_with_fractional_time_windows(self):
        from single_integrator.environment import Config, GiveWayEnv
        for dt in (.05, .07, .1):
            plant = Config(dt=dt, progress_window_seconds=.31, deadlock_hold_seconds=1.13,
                           terminate_on_deadlock=True)
            env = GiveWayEnv(plant)
            env.reset(np.array([[-.6, 0.], [.6, 0.]]))
            before, after, controls = [], [], []
            for _ in range(plant.max_steps):
                before.append(env.positions.copy())
                u = np.zeros((2, 2))
                _, _, done, info = env.step(u)
                after.append(env.positions.copy())
                controls.append(u)
                if done:
                    break
            self.assertTrue(info['deadlock'])
            kwargs = dict(dt=dt, max_speed=plant.max_speed, goal_tolerance=plant.goal_tolerance,
                hold_seconds=plant.deadlock_hold_seconds, progress_window_seconds=plant.progress_window_seconds,
                progress_epsilon=plant.progress_epsilon, speed_epsilon_fraction=plant.speed_epsilon_fraction)
            def score(count):
                return trajectory(jnp.asarray(before[:count]), jnp.asarray(after[:count]),
                    jnp.asarray(controls[:count]), jnp.asarray(env.goals), jnp.ones(count, bool), **kwargs)
            self.assertEqual(int(score(len(after)-1)['complete_windows']), 0)
            r = score(len(after))
            self.assertEqual(int(r['complete_windows']), 1)
            self.assertGreaterEqual(float(r['J_live']), 1.)

    def test_no_deadlock_does_not_imply_task_completion(self):
        # Scene-independent counterexample: separated robots keep circling,
        # never reaching distant goals. This is timeout/livelock, not the
        # specified low-speed deadlock. Do not claim a completion theorem.
        dt = .05
        t = np.arange(851)*dt
        circle = .1*np.c_[np.cos(3*t), np.sin(3*t)]
        positions = np.stack([circle+[-1., 0.], circle+[1., 0.]], axis=1)
        goals = jnp.array([[-5., 0.], [5., 0.]])
        u = np.diff(positions, axis=0)/dt
        result = trajectory(jnp.asarray(positions[:-1]), jnp.asarray(positions[1:]),
            jnp.asarray(u), goals, jnp.ones(850, bool), dt=dt, max_speed=.5,
            goal_tolerance=.08, hold_seconds=5., progress_window_seconds=2.,
            progress_epsilon=.01, speed_epsilon_fraction=.05)
        self.assertGreater(float(jnp.min(jnp.linalg.norm(positions-goals, axis=-1))), 3.)
        self.assertEqual(float(result['J_live']), 0.)


if __name__ == '__main__':
    unittest.main()
