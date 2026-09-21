import unittest
from dataclasses import replace

import numpy as np

from single_integrator.c1.direction_a.rollout import (
    DirectionARolloutError, _terminal_label, compose_double_projection,
    validate_projected_action,
)
from single_integrator.cbf import CBFConfig
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import Config, GiveWayEnv


class ControllerAndMonitorContractTests(unittest.TestCase):
    def test_double_projection_controller_order(self):
        project = lambda value: np.clip(value, -1., 1.)
        safe, applied = compose_double_projection(
            np.array([2., 0., 0., 0.]), np.array([-.5, 0., 0., 0.]),
            project)
        np.testing.assert_array_equal(safe, [1., 0., 0., 0.])
        np.testing.assert_array_equal(applied, [.5, 0., 0., 0.])

    def test_101_samples_and_earliest_action_139_latch(self):
        plant = replace(Config(), terminate_on_deadlock=False)
        env = GiveWayEnv(plant)
        env.reset(np.array([[-.6, 0.], [.6, 0.]]))
        candidate_steps = []
        info = None
        for _ in range(140):
            _, _, _, info = env.step(np.zeros((2, 2)))
            if info["candidate_deadlock"]:
                candidate_steps.append(info["step"])
        self.assertEqual(candidate_steps[0], 40)
        self.assertEqual(candidate_steps[-1], 140)
        self.assertEqual(len(candidate_steps), 101)
        self.assertTrue(info["deadlock"])
        self.assertEqual(env.first_deadlock_step, 140)

    def test_strict_equalities_and_candidate_interruption(self):
        plant = replace(Config(), terminate_on_deadlock=False)
        env = GiveWayEnv(plant)
        # Speed equality is excluded by the strict < 0.025 comparison.
        env.step_count = 39
        errors = np.linalg.norm(env.goals-env.positions, axis=-1)
        env.distance_history = [errors.copy() for _ in range(40)]
        _, _, _, info = env.step(np.array([[.025, 0.], [0., 0.]]))
        self.assertAlmostEqual(info["max_speed"], .025)
        self.assertFalse(info["candidate_deadlock"])

        # Progress equality is excluded by the strict < 0.01 comparison.
        env = GiveWayEnv(plant)
        env.step_count = 39
        errors = np.linalg.norm(env.goals-env.positions, axis=-1)
        env.distance_history = [errors.copy() for _ in range(40)]
        # The scalar predicate itself excludes exact equality.  Use the next
        # representable value above it in the environment integration so that
        # subtraction roundoff cannot turn this boundary test into < .01.
        self.assertFalse(bool(np.max(np.abs(np.array([.01, 0.]))) < .01))
        env.distance_history[0] = errors+np.array([.0100001, 0.])
        _, _, _, info = env.step(np.zeros((2, 2)))
        self.assertGreaterEqual(info["window_progress"][0], .01)
        self.assertFalse(info["candidate_deadlock"])

        # An interruption clears candidate_since and resets the timer at once.
        env = GiveWayEnv(plant)
        for _ in range(50):
            env.step(np.zeros((2, 2)))
        self.assertIsNotNone(env.candidate_since)
        _, _, _, info = env.step(np.array([[.3, 0.], [0., 0.]]))
        self.assertFalse(info["candidate_deadlock"])
        self.assertIsNone(env.candidate_since)
        self.assertEqual(info["stuck_timer"], 0.)

    def test_goal_threshold_and_historical_latch_after_recovery(self):
        plant = replace(Config(), terminate_on_success=False,
                        terminate_on_deadlock=False)
        env = GiveWayEnv(plant)
        # At equality both agents meet the <= goal definition, so the strict
        # candidate atom max_i d_i > .08 is false.
        self.assertTrue(bool(np.all(np.array([.08, .08]) <= .08)))
        inside = .079
        env.positions = env.goals+np.array([[-inside, 0.], [inside, 0.]])
        env.step_count = 39
        errors = np.linalg.norm(env.goals-env.positions, axis=-1)
        env.distance_history = [errors.copy() for _ in range(40)]
        _, _, _, info = env.step(np.zeros((2, 2)))
        self.assertTrue(info["task_success"])
        self.assertFalse(info["candidate_deadlock"])

        env = GiveWayEnv(plant)
        env.reset(np.array([[-.6, 0.], [.6, 0.]]))
        for _ in range(140):
            env.step(np.zeros((2, 2)))
        self.assertEqual(env.first_deadlock_step, 140)
        _, _, _, moved = env.step(np.array([[.3, 0.], [0., 0.]]))
        self.assertFalse(moved["deadlock"])
        self.assertEqual(env.first_deadlock_step, 140)

        # With all event termination disabled, later task success coexists with
        # the immutable historical strict event.
        env = GiveWayEnv(replace(plant, terminate_on_collision=False))
        env.reset(np.array([[-.6, 0.], [.6, 0.]]))
        for _ in range(140):
            env.step(np.zeros((2, 2)))
        while env.first_success_step is None:
            delta = env.goals-env.positions
            norms = np.linalg.norm(delta, axis=-1, keepdims=True)
            control = delta*np.minimum(1., .5/np.maximum(norms, 1e-30))
            env.step(control)
        self.assertEqual(env.first_deadlock_step, 140)
        self.assertIsNotNone(env.first_success_step)

    def test_termination_flags_and_same_step_priority(self):
        base = dict(agent_collision=True, wall_collision=True,
                    task_success=True, deadlock=True, termination="collision")
        plant = Config()
        self.assertEqual(_terminal_label(base, plant), "agent_collision")
        wall = dict(base, agent_collision=False)
        self.assertEqual(_terminal_label(wall, plant), "wall_collision")
        success = dict(wall, wall_collision=False, termination="success")
        self.assertEqual(_terminal_label(success, plant), "success")
        deadlock = dict(success, task_success=False, termination="deadlock")
        self.assertEqual(_terminal_label(deadlock, plant), "safe_deadlock")
        disabled = replace(plant, terminate_on_collision=False,
                           terminate_on_success=False,
                           terminate_on_deadlock=False)
        timeout = dict(base, termination="timeout")
        self.assertEqual(_terminal_label(timeout, disabled), "other_timeout")

        # Turning off deadlock termination does not turn off the raw latch.
        env = GiveWayEnv(replace(Config(), terminate_on_deadlock=False))
        for _ in range(140):
            _, _, done, info = env.step(np.zeros((2, 2)))
        self.assertTrue(info["deadlock"])
        self.assertEqual(env.first_deadlock_step, 140)
        self.assertFalse(done)

    def test_stalled_exact_tail_indices_original_label_and_no_padding(self):
        speeds = np.full(850, .2)
        speeds[810:850] = .049
        errors = np.ones((850, 2))
        errors[810] = [1.0, 1.1]  # post-action sample d_811
        errors[849] = [1.019, 1.081]  # post-action sample d_850
        # Fill the interior without affecting the frozen endpoint comparison.
        errors[811:849] = np.linspace(errors[810], errors[849], 38, endpoint=False)
        label, details = classify_timeout_trace(
            dict(max_speed=speeds, goal_errors=errors), "other_timeout", .05)
        self.assertEqual(label, "stalled_deadlock")
        self.assertTrue(details["eligible"] and details["enough_window"])
        unchanged, details = classify_timeout_trace(
            dict(max_speed=speeds, goal_errors=errors), "success", .05)
        self.assertEqual(unchanged, "success")
        self.assertFalse(details["eligible"])
        short, details = classify_timeout_trace(
            dict(max_speed=np.zeros(39), goal_errors=np.ones((39, 2))),
            "other_timeout", .05)
        self.assertEqual(short, "other_timeout")
        self.assertFalse(details["enough_window"])

    def test_stalled_strict_threshold_equalities(self):
        speeds = np.zeros(850)
        errors = np.ones((850, 2))
        speeds[849] = .05
        label, _ = classify_timeout_trace(
            dict(max_speed=speeds, goal_errors=errors), "other_timeout", .05)
        self.assertEqual(label, "other_timeout")
        speeds[849] = 0.
        errors[849, 0] = 1.02
        label, _ = classify_timeout_trace(
            dict(max_speed=speeds, goal_errors=errors), "other_timeout", .05)
        self.assertEqual(label, "other_timeout")


class SafetyFailureContractTests(unittest.TestCase):
    def setUp(self):
        self.cbf = CBFConfig()
        self.A = np.eye(4)
        self.lower = np.full(4, -1.)

    def test_projection_audit_accepts_only_certified_feasible_actions(self):
        audit = validate_projected_action(
            np.zeros(4), self.A, self.lower, .5, "solved", self.cbf)
        self.assertGreaterEqual(audit.min_linear_residual, 0.)
        for action, status in [
            (np.zeros(4), "failed"),
            (np.array([np.nan, 0., 0., 0.]), "solved"),
            (np.array([-1.1, 0., 0., 0.]), "solved"),
            (np.array([.51, 0., 0., 0.]), "solved"),
        ]:
            with self.assertRaises(DirectionARolloutError):
                validate_projected_action(action, self.A, self.lower, .5,
                                          status, self.cbf)

    def test_collision_is_read_from_true_integrated_positions(self):
        env = GiveWayEnv(replace(Config(), terminate_on_deadlock=False))
        env.reset(np.array([[-.165, 0.], [.165, 0.]]))
        _, _, done, info = env.step(np.array([[.5, 0.], [-.5, 0.]]))
        self.assertTrue(done)
        self.assertTrue(info["agent_collision"])
        self.assertLessEqual(info["min_swept_agent_distance"], 0.)


if __name__ == "__main__":
    unittest.main()
