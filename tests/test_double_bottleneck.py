"""Small engineering/regression tests for the four-agent scenario.

These tests deliberately avoid eta sweeps.  They exercise plant semantics,
event precedence, controller plumbing, hard projection, and visualization.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from double_bottleneck import Config, DoubleBottleneckEnv, INITIAL_REGIMES
from double_bottleneck.controller import DoubleBottleneckController
from double_bottleneck.rollout import run_rollout
from double_bottleneck.smoke import PlumbingGoalPairPolicy, run_smoke
from double_bottleneck.visualization import save_trajectory_svg
from shared_control.hard_projection import HardSafetyFilter
from shared_control.diagnostic_corrector import all_pair_mean_relative_basis, bounded_rows


class DoubleBottleneckEnvironmentTests(unittest.TestCase):
    def test_all_four_agents_initialize_in_all_regimes(self):
        for regime in INITIAL_REGIMES:
            with self.subTest(regime=regime):
                env = DoubleBottleneckEnv(Config(initial_regime=regime))
                self.assertEqual(env.positions.shape, (4, 2))
                self.assertEqual(env.goals.shape, (4, 2))
                self.assertEqual(env.observation().shape, (4, 18))
                self.assertEqual(env.all_pair_observations().shape, (6, 2, 10))
                self.assertFalse(env.outside(env.positions).any())
                self.assertGreater(env.distances()[0].min(), env.config.wall_collision_margin)
                self.assertGreater(env.distances()[1].min(), env.config.agent_collision_margin)

    def test_action_shape_speed_bound_and_exact_dynamics(self):
        env = DoubleBottleneckEnv()
        before = env.positions.copy()
        action = np.array(((0.1, 0.0), (0.0, -0.1), (-0.2, 0.0), (0.0, 0.2)))
        _, _, done, info = env.step(action)
        self.assertFalse(done)
        np.testing.assert_array_equal(env.positions, before + env.config.dt * action)
        np.testing.assert_array_equal(env.velocities, action)
        self.assertLessEqual(info["integration_residual"], 5e-16)
        with self.assertRaises(ValueError):
            DoubleBottleneckEnv().step(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            DoubleBottleneckEnv().step(np.array(((0.51, 0.0),) * 4))

    def test_wall_collision_is_observed_without_hidden_response(self):
        env = DoubleBottleneckEnv()
        positions = env.positions.copy()
        positions[0, 1] = env.config.outer_half_height - env.config.agent_radius - 0.02
        env.reset(positions)
        before = env.positions.copy()
        action = np.zeros((4, 2))
        action[0, 1] = env.config.max_speed
        _, _, done, info = env.step(action)
        self.assertTrue(done)
        self.assertEqual(info["termination"], "collision")
        self.assertTrue(info["wall_collision"])
        np.testing.assert_array_equal(env.positions, before + env.config.dt * action)

    def test_agent_collision_checks_all_six_pairs(self):
        env = DoubleBottleneckEnv(Config(dt=0.2))
        positions = np.array(((-0.2, -0.25), (-0.2, 0.45), (0.2, -0.25), (0.2, 0.45)))
        env.reset(positions)
        before = env.positions.copy()
        action = np.array(((0.5, 0.0), (0.0, 0.0), (-0.5, 0.0), (0.0, 0.0)))
        _, _, done, info = env.step(action)
        self.assertTrue(done)
        self.assertEqual(info["termination"], "collision")
        self.assertTrue(info["agent_collision"])
        self.assertEqual(info["agent_surface_distances"].shape, (6,))
        self.assertEqual(info["swept_agent_distances"].shape, (6,))
        np.testing.assert_array_equal(env.positions, before + env.config.dt * action)

    def test_success_requires_all_four_agents(self):
        almost = DoubleBottleneckEnv()
        positions = almost.goals.copy()
        positions[3, 0] += 0.10  # Outside the 0.08 goal tolerance, still in workspace.
        almost.reset(positions)
        _, _, done, info = almost.step(np.zeros((4, 2)))
        self.assertFalse(done)
        self.assertFalse(info["task_success"])

        complete = DoubleBottleneckEnv()
        complete.reset(complete.goals.copy())
        _, _, done, info = complete.step(np.zeros((4, 2)))
        self.assertTrue(done)
        self.assertEqual(info["termination"], "success")
        self.assertTrue(complete.summary()["final_goal_success"])

    def test_timeout_remains_separate_from_deadlock(self):
        env = DoubleBottleneckEnv(Config(max_steps=3, terminate_on_deadlock=False))
        for index in range(3):
            _, _, done, info = env.step(np.zeros((4, 2)))
            self.assertEqual(done, index == 2)
        self.assertEqual(info["termination"], "timeout")
        self.assertFalse(info["deadlock"])

    def test_global_strict_deadlock_monitor_keeps_legacy_timing(self):
        env = DoubleBottleneckEnv()
        for index in range(140):
            _, _, done, info = env.step(np.zeros((4, 2)))
            self.assertEqual(done, index == 139)
        self.assertEqual(info["termination"], "deadlock")
        self.assertEqual(env.first_deadlock_step, 140)
        self.assertAlmostEqual(info["time"], 7.0)

    def test_partial_stall_is_not_global_strict_deadlock_candidate(self):
        """Documents, rather than silently changing, legacy global semantics.

        Three agents are stationary while one makes measurable progress.  The
        global max-progress/max-speed predicate is false, so this is not strict
        deadlock even though a per-agent monitor could call it partial stall.
        """
        env = DoubleBottleneckEnv()
        action = np.zeros((4, 2))
        action[0, 0] = 0.05
        for _ in range(40):
            _, _, done, info = env.step(action)
            self.assertFalse(done)
        self.assertTrue(info["window_ready"])
        self.assertFalse(info["candidate_deadlock"])
        np.testing.assert_allclose(info["window_progress"][1:], 0.0, atol=1e-14)
        self.assertGreater(info["window_progress"][0], env.config.progress_epsilon)

    def test_augmented_state_restores_monitor_memory_exactly(self):
        env = DoubleBottleneckEnv()
        for _ in range(17):
            env.step(np.zeros((4, 2)))
        state = env.augmented_state()
        restored = DoubleBottleneckEnv()
        observation = restored.restore_augmented_state(state)
        np.testing.assert_array_equal(observation, env.observation())
        np.testing.assert_array_equal(restored.positions, env.positions)
        np.testing.assert_array_equal(restored.velocities, env.velocities)
        np.testing.assert_array_equal(restored.distance_history, env.distance_history)
        self.assertEqual(restored.step_count, env.step_count)
        self.assertEqual(restored.candidate_since, env.candidate_since)


class DoubleBottleneckVisualizationTests(unittest.TestCase):
    def test_static_svg_contains_scene_eta_and_terminal_reason(self):
        env = DoubleBottleneckEnv()
        positions = np.stack((env.positions, env.positions + np.array(((0.1, 0), (0.1, 0), (-0.1, 0), (-0.1, 0)))))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectory.svg"
            returned = save_trajectory_svg(env, positions, (1.0, 0.0, 0.25), "timeout", output)
            self.assertEqual(returned, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("fixed eta=(1, 0, 0.25)", text)
            self.assertIn("terminal=timeout", text)
            self.assertEqual(text.count("<polyline"), 4)
            for name in ("A1", "A2", "B1", "B2"):
                self.assertIn(name, text)


class DoubleBottleneckProjectionTests(unittest.TestCase):
    def test_global_hard_projection_has_six_pairs_and_four_speed_cones(self):
        env = DoubleBottleneckEnv()
        target = np.array(((2.0, 0.2), (1.0, -1.0), (-2.0, 0.2), (-1.0, -1.0)))
        result = HardSafetyFilter()(env.snapshot(), target)
        self.assertEqual(result.velocity.shape, (4, 2))
        self.assertEqual(result.diagnostics["num_agents"], 4)
        self.assertEqual(result.diagnostics["num_pair_constraints"], 6)
        self.assertEqual(result.diagnostics["num_wall_constraints"], 4 * len(env.walls))
        self.assertTrue(result.diagnostics["accepted"])
        self.assertLessEqual(np.linalg.norm(result.velocity, axis=-1).max(), env.config.max_speed + 1e-9)
        self.assertGreaterEqual(result.diagnostics["min_linear_residual"], -1e-9)

        # The certified held velocity must advance the plant without a first-step
        # wall or inter-agent collision.
        _, _, _, info = env.step(result.velocity)
        self.assertFalse(info["wall_collision"])
        self.assertFalse(info["agent_collision"])

    def test_projection_blocks_an_imminent_pair_collision(self):
        env = DoubleBottleneckEnv(Config(dt=0.2))
        env.reset(np.array(((-0.2, -0.25), (-0.2, 0.45), (0.2, -0.25), (0.2, 0.45))))
        desired = np.array(((0.5, 0.0), (0.0, 0.0), (-0.5, 0.0), (0.0, 0.0)))
        result = HardSafetyFilter()(env.snapshot(), desired)
        self.assertFalse(np.allclose(result.velocity, desired))
        self.assertGreaterEqual(result.diagnostics["min_linear_residual"], -1e-9)
        _, _, _, info = env.step(result.velocity)
        self.assertFalse(info["agent_collision"])
        self.assertGreater(info["min_swept_agent_distance"], env.config.agent_collision_margin)

    def test_projection_blocks_an_imminent_wall_collision(self):
        env = DoubleBottleneckEnv(Config(dt=0.2))
        positions = env.positions.copy()
        positions[0, 1] = env.config.outer_half_height - env.config.agent_radius - 0.02
        env.reset(positions)
        desired = np.zeros((4, 2))
        desired[0, 1] = env.config.max_speed
        result = HardSafetyFilter()(env.snapshot(), desired)
        self.assertLess(result.velocity[0, 1], desired[0, 1])
        _, _, _, info = env.step(result.velocity)
        self.assertFalse(info["wall_collision"])
        self.assertGreater(info["min_swept_wall_distance"], env.config.wall_collision_margin)


class DoubleBottleneckControllerTests(unittest.TestCase):
    def test_closed_loop_controller_runs_both_global_projections(self):
        import jax

        env = DoubleBottleneckEnv()
        eta = (0.7, -0.2, 0.15)
        result = DoubleBottleneckController(PlumbingGoalPairPolicy()).action(
            env, eta, jax.random.PRNGKey(13), step=0
        )
        for value in (
            result.u_flow,
            result.u_safe,
            result.b_goal,
            result.b_rel,
            result.g_raw,
            result.w,
            result.u_exec,
        ):
            self.assertEqual(value.shape, (4, 2))
            self.assertTrue(np.isfinite(value).all())
        np.testing.assert_allclose(
            result.g_raw,
            eta[0] * result.b_goal + eta[1] * result.u_safe + eta[2] * result.b_rel,
            rtol=0,
            atol=2e-16,
        )
        np.testing.assert_array_equal(result.w, result.u_safe + result.g_raw)
        for projection in (result.first_projection, result.second_projection):
            self.assertTrue(projection["accepted"])
            self.assertEqual(projection["num_agents"], 4)
            self.assertEqual(projection["num_pair_constraints"], 6)
        self.assertLessEqual(np.linalg.norm(result.u_exec, axis=-1).max(), env.config.max_speed + 1e-9)

    def test_fixed_eta_is_recomputed_through_entire_smoke_rollout(self):
        eta = (0.25, -0.1, 0.05)
        result = run_rollout(
            PlumbingGoalPairPolicy(), eta, regime="clearly_asymmetric", seed=5, max_steps=4
        )
        self.assertEqual(result.eta, eta)
        self.assertEqual(result.summary["termination"], "timeout")
        self.assertEqual(len(result.records), 4)
        self.assertEqual(result.positions.shape, (5, 4, 2))
        for record in result.records:
            control = record.control
            np.testing.assert_allclose(
                control.g_raw,
                eta[0] * control.b_goal + eta[1] * control.u_safe + eta[2] * control.b_rel,
                rtol=0,
                atol=2e-16,
            )
        # Nonconstant bases demonstrate statewise recomputation, rather than a
        # fixed open-loop correction vector.
        self.assertGreater(np.linalg.norm(result.records[-1].control.g_raw - result.records[0].control.g_raw), 0)

    def test_checkpoint_free_smoke_reports_non_scientific_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            svg = Path(directory) / "smoke.svg"
            result = run_smoke((1.0, 0.0, 0.25), "near_symmetric", 2, 3, svg)
            self.assertFalse(result["scientific_policy"])
            self.assertEqual(result["executed_steps"], 3)
            self.assertEqual(result["positions_shape"], [4, 4, 2])
            self.assertEqual(result["action_shape"], [4, 2])
            self.assertEqual(result["termination"], "timeout")
            self.assertTrue(result["both_projections_observed"])
            self.assertTrue(svg.is_file())


class FourAgentRelationBasisTests(unittest.TestCase):
    def test_four_agent_basis_is_bound_then_all_pair_mean(self):
        positions = DoubleBottleneckEnv().positions
        basis = all_pair_mean_relative_basis(positions, max_speed=0.5)
        for agent in range(4):
            expected = bounded_rows(
                positions[agent] - np.delete(positions, agent, axis=0), 0.5
            ).mean(axis=0)
            np.testing.assert_allclose(basis[agent], expected, rtol=0, atol=1e-16)

    def test_relation_basis_reduces_exactly_to_single_opponent_at_n2(self):
        positions = np.array(((-1.0, 0.1), (0.4, -0.2)))
        expected = bounded_rows(
            np.array((positions[0] - positions[1], positions[1] - positions[0])), 0.5
        )
        np.testing.assert_array_equal(all_pair_mean_relative_basis(positions, 0.5), expected)

    def test_relation_basis_is_permutation_equivariant(self):
        positions = DoubleBottleneckEnv().positions
        permutation = np.array((2, 0, 3, 1))
        original = all_pair_mean_relative_basis(positions, 0.5)
        permuted = all_pair_mean_relative_basis(positions[permutation], 0.5)
        np.testing.assert_allclose(permuted, original[permutation], rtol=0, atol=1e-16)


if __name__ == "__main__":
    unittest.main()
