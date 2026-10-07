"""Focused Gate-A tests for the centralized four-agent expert."""

from __future__ import annotations

import unittest

import numpy as np

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.expert import (
    CentralizedExpert,
    CoordinationHypothesis,
    all_coordination_hypotheses,
)
from double_bottleneck.scenario import INITIAL_REGIMES


class CentralizedExpertTests(unittest.TestCase):
    def test_all_regimes_pass_default_eight_hypothesis_search(self):
        planner = CentralizedExpert()
        for regime in INITIAL_REGIMES:
            with self.subTest(regime=regime):
                env = DoubleBottleneckEnv(Config(initial_regime=regime))
                plan = planner.plan(env)
                self.assertTrue(plan.success)
                self.assertEqual(plan.terminal_reason, "success")
                self.assertFalse(plan.collision)
                self.assertFalse(plan.timeout)
                self.assertFalse(plan.deadlock)
                self.assertEqual(plan.candidate_count, 8)
                self.assertLess(plan.episode_steps, env.config.max_steps)
                self.assertGreater(
                    plan.min_inter_agent_surface_distance,
                    env.config.agent_collision_margin,
                )
                self.assertGreater(
                    plan.min_wall_clearance,
                    env.config.wall_collision_margin,
                )
                self.assertEqual(plan.actions.shape, (plan.episode_steps, 4, 2))
                self.assertEqual(plan.positions.shape, (plan.episode_steps + 1, 4, 2))
                self.assertEqual(
                    plan.observations.shape, (plan.episode_steps + 1, 4, 18)
                )
                self.assertLessEqual(
                    np.linalg.norm(plan.actions, axis=-1).max(),
                    env.config.max_speed + 1e-9,
                )
                np.testing.assert_allclose(
                    np.diff(plan.positions, axis=0),
                    env.config.dt * plan.actions,
                    rtol=0,
                    atol=2e-15,
                )

    def test_near_symmetric_state_supports_both_direction_first_modes(self):
        env = DoubleBottleneckEnv(Config(initial_regime="near_symmetric"))
        planner = CentralizedExpert()
        left_first = planner.plan_hypothesis(
            env,
            CoordinationHypothesis("left_to_right", (0, 1), (2, 3)),
        )
        right_first = planner.plan_hypothesis(
            env,
            CoordinationHypothesis("right_to_left", (0, 1), (2, 3)),
        )
        self.assertTrue(left_first.success)
        self.assertTrue(right_first.success)
        self.assertNotEqual(
            left_first.hypothesis.first_direction,
            right_first.hypothesis.first_direction,
        )
        # These are qualitatively different schedules, not numerical noise:
        # at 8 s the first-wave populations occupy opposite sides/chambers.
        index = int(8.0 / env.config.dt)
        self.assertGreater(
            np.linalg.norm(left_first.positions[index] - right_first.positions[index]),
            1.0,
        )

    def test_all_eight_coordination_hypotheses_are_unique(self):
        seeds = all_coordination_hypotheses()
        self.assertEqual(len(seeds), 8)
        self.assertEqual(len({seed.label for seed in seeds}), 8)
        self.assertEqual(
            {seed.first_direction for seed in seeds}, {"left_to_right", "right_to_left"}
        )

    def test_initial_last_applied_velocity_is_preserved_in_dataset_state(self):
        env = DoubleBottleneckEnv(Config(initial_regime="weakly_asymmetric"))
        initial_velocity = np.array(
            ((0.08, 0.01), (0.03, -0.02), (-0.07, 0.01), (-0.04, -0.01)),
            dtype=np.float64,
        )
        env.velocities = initial_velocity.copy()
        plan = CentralizedExpert().plan_hypothesis(
            env,
            CoordinationHypothesis("left_to_right", (0, 1), (2, 3)),
        )
        self.assertTrue(plan.success)
        np.testing.assert_allclose(plan.observations[0, :, 2:4], initial_velocity)
        # Single-integrator semantics: the first action, not the previous
        # reported velocity, determines the first position increment.
        np.testing.assert_allclose(
            plan.positions[1] - plan.positions[0],
            env.config.dt * plan.actions[0],
            rtol=0,
            atol=2e-15,
        )

    def test_schedule_permits_temporary_motion_away_from_goal_for_yielding(self):
        env = DoubleBottleneckEnv(Config(initial_regime="weakly_asymmetric"))
        plan = CentralizedExpert().plan_hypothesis(
            env,
            CoordinationHypothesis("left_to_right", (0, 1), (2, 3)),
        )
        goal_errors = np.linalg.norm(env.goals[None, :, :] - plan.positions, axis=-1)
        self.assertTrue((np.diff(goal_errors, axis=0) > 1e-6).any())


if __name__ == "__main__":
    unittest.main()
