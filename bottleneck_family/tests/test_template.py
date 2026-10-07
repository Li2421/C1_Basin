import json
import unittest
from dataclasses import replace

import numpy as np

from bottleneck_family import BottleneckEnv, Config, build_instance
from bottleneck_family.__main__ import preset
from bottleneck_family.scenario import Geometry


class TemplateTests(unittest.TestCase):
    def test_requested_sizes_share_bottlenecks_and_safe_geometry(self):
        for n in (2, 4, 10, 20, 50, 100):
            with self.subTest(n=n):
                env = BottleneckEnv(Config(num_agents=n))
                self.assertEqual(env.positions.shape, (n,2))
                self.assertEqual(env.observation()['agents'].shape, (n,6))
                self.assertEqual(len(env.pair_indices), n*(n-1)//2)
                report = env.instance.manifest()['validation']
                self.assertEqual(report['opposing_pairs_sharing_barriers'], (n//2)*((n+1)//2))
                self.assertEqual(report['joint_feasibility'], 'not_certified')
                self.assertTrue(np.all(env.positions[:,0]*env.goals[:,0] < 0))
                _, done, info = env.step(np.zeros((n,2)))
                self.assertFalse(done)
                self.assertGreater(info['swept_clearance'], 0)
                for path in env.instance.paths:
                    self.assertTrue(all(env.instance.geometry.visible(a,b) for a,b in zip(path,path[1:])))

    def test_size_sweep_keeps_existing_tasks(self):
        largest = build_instance(Config(num_agents=100))
        for n in (2, 10, 20, 50):
            smaller = build_instance(Config(num_agents=n))
            np.testing.assert_array_equal(smaller.positions, largest.positions[:n])
            np.testing.assert_array_equal(smaller.goals, largest.goals[:n])

    def test_obstacle_variants_at_100_agents(self):
        fingerprints = set()
        for name in ('gap','double','offset','wide','three_doors','clutter'):
            instance = build_instance(preset(name, 100))
            self.assertEqual(len(instance.paths), 100)
            fingerprints.add(instance.config.fingerprint)
        self.assertEqual(len(fingerprints), 6)

    def test_invalid_geometry_and_insufficient_capacity_rejected(self):
        for kwargs in ({'openings': (((0.,.2),),)}, {'num_agents': 1},
                       {'barrier_x': (1.,-1.), 'openings': (((0.,.5),),((0.,.5),))},
                       {'extra_rectangles': ((0.,0.,-1.,1.),)}, {'dt': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            build_instance(Config(num_agents=100, half_length=2., half_height=1.))

    def test_blocked_door_rejected_as_infeasible_not_deadlock(self):
        with self.assertRaisesRegex(ValueError, 'path witness'):
            build_instance(Config(extra_rectangles=((-0.4,-.4,.4,.4),)))

    def test_repeatability_split_and_json_roundtrip(self):
        c = Config(num_agents=20)
        first, second = build_instance(c), build_instance(Config(**json.loads(json.dumps(c.to_dict()))))
        np.testing.assert_array_equal(first.positions, second.positions)
        self.assertEqual(c.fingerprint, second.config.fingerprint)
        other = build_instance(replace(c, split='test'))
        self.assertFalse(np.array_equal(first.positions, other.positions))

    def test_wall_tunneling_detected(self):
        env = BottleneckEnv(Config(num_agents=2, dt=1., max_speed=3.))
        env.reset([[-1.,1.],[5.,2.]], [[5.,1.],[-5.,2.]])
        _,done,info = env.step([[2.,0.],[0.,0.]])
        self.assertTrue(done)
        self.assertEqual(info['termination'], 'collision')
        self.assertLess(info['min_swept_wall_clearance'], 0)

    def test_agent_tunneling_detected(self):
        env = BottleneckEnv(Config(num_agents=2, dt=1., max_speed=3.))
        env.reset([[-4.,1.],[-2.,1.]], [[5.,1.],[5.,2.]])
        _,done,info = env.step([[2.,0.],[-2.,0.]])
        self.assertTrue(done)
        self.assertLess(info['min_swept_agent_clearance'],0)

    def test_partial_stall_not_hidden_by_moving_agent(self):
        env = BottleneckEnv(Config(num_agents=3))
        env.reset([[-3.,0.],[-2.5,0.],[3.,2.]], [[3.,0.],[3.5,0.],[-3.,2.]])
        for _ in range(101):
            _,_,info = env.step([[0.,0.],[0.,0.],[0.,-.05]])
        self.assertIn([0,1], info['stalled_groups'])
        self.assertNotIn('deadlock', info)
        self.assertFalse(env.done)

    def test_goal_success_timeout_and_post_termination_guard(self):
        env = BottleneckEnv(Config(num_agents=2, max_steps=1))
        _,done,info = env.step(np.zeros((2,2)))
        self.assertTrue(done)
        self.assertEqual(info['termination'], 'timeout')
        with self.assertRaises(RuntimeError):
            env.step(np.zeros((2,2)))
        p = env.instance.positions
        env.reset(p, p)
        _,done,info = env.step(np.zeros((2,2)))
        self.assertEqual(info['termination'], 'success')
        self.assertTrue(info['task_success'])

    def test_shared_safety_filter_supports_100_agent_snapshot(self):
        try:
            from shared_control.hard_projection import HardSafetyFilter
        except ImportError:
            self.skipTest('shared safety integration requires scipy and clarabel')
        env = BottleneckEnv(Config(num_agents=100))
        result = HardSafetyFilter()(env.snapshot(), np.zeros((100,2)))
        self.assertEqual(result.velocity.shape, (100,2))
        self.assertEqual(result.diagnostics['num_pair_constraints'], 4950)
        self.assertTrue(result.diagnostics['accepted'])

    def test_reference_expert_proves_joint_feasibility_at_requested_sizes(self):
        from bottleneck_family.expert import SequentialExpert
        planner = SequentialExpert()
        for n in (2, 10, 50):
            with self.subTest(n=n):
                env = BottleneckEnv(Config(num_agents=n, max_steps=max(2000,n*800)))
                result = planner.rollout(env)
                self.assertTrue(result['success'], result.get('reason'))
                self.assertEqual(result['reason'], 'success')
                self.assertEqual(result['states'].shape[0], len(result['actions'])+1)
                self.assertGreater(result['min_clearance'], 0)
                self.assertTrue(np.all(np.linalg.norm(env.positions-env.goals,axis=1)
                                       <= env.config.goal_tolerance))

    def test_inside_rectangle_is_invalid(self):
        geometry = Geometry(Config())
        self.assertFalse(geometry.valid_points(np.array([[0.,2.]]))[0])


if __name__ == '__main__':
    unittest.main()
