import math
import unittest
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.vi_r_cert import (
    STRICT_AUGMENTED_COUNT, STRICT_DIRECT_COUNT,
    STALLED_AUGMENTED_COUNT, STALLED_DIRECT_COUNT,
    _vi_margins, squared_distance_goal_bad_set,
    squared_distance_progress_bad_set, squared_distance_speed_bad_set,
    trajectory)
from single_integrator.c1.termination import (
    EVENTS, _enabled_terminal_code, _monitor_oracle)
from single_integrator.environment import Config, GiveWayEnv


jax.config.update('jax_enable_x64', True)


class BadSetDistanceTests(unittest.TestCase):
    def test_closed_form_distances(self):
        q = jnp.array([0., 0., .1, 0.])
        self.assertAlmostEqual(float(squared_distance_speed_bad_set(q, .05)),
                               .05 ** 2)
        positions = jnp.zeros((2, 2))
        goals = jnp.array([[.1, 0.], [-.1, 0.]])
        # q=0 leaves both agents .1m from goal; distance to either exterior
        # boundary of the .08m goal ball is (.08-.1)_+=0.
        self.assertEqual(float(squared_distance_goal_bad_set(
            jnp.zeros(4), positions, goals, .05, .08)), 0.)
        # Anchor .1m with eps .01 gives a control-space annulus [1.8,2.2]
        # around a zero center.
        d = squared_distance_progress_bad_set(
            jnp.zeros(4), positions, jnp.zeros((2, 2)),
            jnp.array([.1, .1]), .05, .01)
        self.assertAlmostEqual(float(d), 2 * 1.8 ** 2, places=12)

    def test_frozen_counts(self):
        self.assertEqual(STRICT_DIRECT_COUNT, 303)
        self.assertEqual(STRICT_AUGMENTED_COUNT, 2727)
        self.assertEqual(STALLED_DIRECT_COUNT, 42)
        self.assertEqual(STALLED_AUGMENTED_COUNT, 378)

    def test_conservative_subspace_counterexample(self):
        # U=[-2,2]x{0}, w=(2,10), B=unit ball.  Four signed coordinate
        # witnesses include the duplicated projection (0,0).
        w = jnp.array([2., 10.])
        v = jnp.array([[.5, 0.], [-.5, 0.], [0., 0.], [0., 0.]])
        values = (_vi_margins(w, v, lambda q:
                  jnp.maximum(jnp.linalg.norm(q, axis=-1) - 1., 0.) ** 2)
                  * .25)  # undo the module's fixed division by .25
        expected = np.array([-8.307764064, -10.111874208,
                             -9.198039027, -9.198039027])
        np.testing.assert_allclose(values, expected, rtol=0, atol=2e-8)
        self.assertTrue(np.all(np.asarray(values) <= 0.))


class GuardedAggregationTests(unittest.TestCase):
    @staticmethod
    def trace_inputs(latch_pre=None):
        goals = jnp.array([[1.09, 0.], [-1.09, 0.]])
        state = jnp.array([[-.4, 0.], [.4, 0.]])
        before = jnp.broadcast_to(state, (850, 2, 2))
        after = before
        w = jnp.zeros((850, 4))
        applied = jnp.zeros((850, 4))
        witnesses = jnp.zeros((850, 8, 4))
        alive = jnp.ones((850,), bool)
        latch = (jnp.zeros((850,), bool) if latch_pre is None
                 else jnp.asarray(latch_pre, bool))
        return before, after, w, applied, witnesses, goals, alive, latch

    def test_strict_event_implies_both_risks_at_least_one(self):
        result = trajectory(*self.trace_inputs(), False)
        self.assertGreaterEqual(float(result['direct_R_CERT']), 1.-1e-12)
        self.assertGreaterEqual(float(result['augmented_R_CERT']), 1.-1e-12)

    def test_historical_guard_snapshot_survives_latch(self):
        latch = np.zeros(850, bool)
        latch[140:] = True
        result = trajectory(*self.trace_inputs(latch), False)
        self.assertTrue(bool(result['strict_guards'][0]))  # t=139
        self.assertFalse(bool(result['strict_guards'][1]))
        self.assertGreaterEqual(float(result['augmented_R_CERT']), 1.-1e-12)

    def test_stalled_timeout_uses_exact_tail_and_no_latch_exclusion(self):
        latch = np.ones(850, bool)
        result = trajectory(*self.trace_inputs(latch), True)
        self.assertTrue(bool(result['stalled_guard']))
        self.assertGreaterEqual(float(result['direct_R_CERT']), 1.-1e-12)
        self.assertGreaterEqual(float(result['augmented_R_CERT']), 1.-1e-12)

    def test_no_valid_window_is_explicitly_undefined(self):
        values = list(self.trace_inputs())
        values[6] = jnp.arange(850) < 100
        values[2] = values[2].at[100:].set(jnp.nan)
        values[3] = values[3].at[100:].set(jnp.nan)
        values[4] = values[4].at[100:].set(jnp.nan)
        # Even an original timeout cannot enable stalled without the complete
        # 850-action tail; no padded tail is manufactured.
        result = trajectory(*values, True)
        self.assertFalse(bool(result['eligible']))
        self.assertFalse(bool(result['stalled_guard']))
        self.assertTrue(math.isnan(float(result['augmented_R_CERT'])))

    def test_stable_margin_finite_difference(self):
        q = jnp.array([.2, 0., .3, 0.])
        f = lambda z: squared_distance_speed_bad_set(z, .05)
        direction = jnp.array([.1, .2, -.1, .3])
        ad = float(jnp.vdot(jax.grad(f)(q), direction))
        for h in (1e-3, 1e-4, 1e-5):
            fd = float((f(q+h*direction)-f(q-h*direction))/(2*h))
            self.assertAlmostEqual(fd, ad, delta=1e-8)


class MonitorSemanticsTests(unittest.TestCase):
    def test_raw_latch_independent_of_deadlock_termination_flag(self):
        positions = np.array([[[-.4, 0.], [.4, 0.]]])
        controls = np.zeros((1, 2, 2))
        goals = GiveWayEnv().goals
        errors = np.linalg.norm(positions[0] - goals, axis=-1)[None]
        for enabled in (True, False):
            plant = replace(Config(), terminate_on_deadlock=enabled)
            result = _monitor_oracle(
                positions, controls, errors, errors, np.array([40]),
                np.array([-1]), np.array([True]), 139, plant)
            code, _, first, raw, candidate, _ = result
            self.assertTrue(raw[0] and candidate[0])
            self.assertEqual(first[0], 140)
            self.assertEqual(code[0], EVENTS['safe_deadlock'] if enabled else 0)

    def test_deadlock_recovery_does_not_clear_first_step(self):
        plant = replace(Config(), terminate_on_deadlock=False)
        positions = np.array([[[-.4, 0.], [.4, 0.]]])
        goals = GiveWayEnv(plant).goals
        errors = np.linalg.norm(positions[0] - goals, axis=-1)[None]
        first = _monitor_oracle(
            positions, np.zeros((1, 2, 2)), errors, errors,
            np.array([40]), np.array([-1]), np.array([True]), 139, plant)
        moved = _monitor_oracle(
            positions, np.array([[[.1, 0.], [0., 0.]]]), errors, errors,
            first[1], first[2], np.array([True]), 140, plant)
        self.assertTrue(first[3][0])
        self.assertFalse(moved[3][0])
        self.assertEqual(moved[2][0], 140)

    def test_last_action_raw_deadlock_can_terminal_label_timeout(self):
        plant = replace(Config(), terminate_on_deadlock=False)
        positions = np.array([[[-.4, 0.], [.4, 0.]]])
        goals = GiveWayEnv(plant).goals
        errors = np.linalg.norm(positions[0] - goals, axis=-1)[None]
        result = _monitor_oracle(
            positions, np.zeros((1, 2, 2)), errors, errors,
            np.array([750]), np.array([-1]), np.array([True]), 849, plant)
        self.assertTrue(result[3][0])
        self.assertEqual(result[2][0], 850)
        self.assertEqual(result[0][0], EVENTS['other_timeout'])

    def test_enabled_priority_and_disabled_timeout(self):
        base = dict(termination='collision', agent_collision=True,
                    wall_collision=True, task_success=True, deadlock=True)
        plant = Config()
        self.assertEqual(_enabled_terminal_code(base, plant),
                         EVENTS['agent_collision'])
        wall = dict(base, agent_collision=False)
        self.assertEqual(_enabled_terminal_code(wall, plant),
                         EVENTS['wall_collision'])
        timeout = dict(base, termination='timeout')
        disabled = replace(plant, terminate_on_collision=False,
                           terminate_on_success=False,
                           terminate_on_deadlock=False)
        self.assertEqual(_enabled_terminal_code(timeout, disabled),
                         EVENTS['other_timeout'])


if __name__ == '__main__':
    unittest.main()
