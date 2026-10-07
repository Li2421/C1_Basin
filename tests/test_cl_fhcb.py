"""Deterministic/unit coverage for the new CL-FHCB construction."""

import unittest

import numpy as np

from diagnostics.cl_fhcb.certificate import (
    BIN_INDEX,
    BIN_NAMES,
    EmpiricalContinuationCertificate,
    abstract_bin,
    score_prefix,
)
from diagnostics.cl_fhcb.closed_loop import (
    AugmentedState,
    ClosedLoopStep,
    ClosedLoopTrace,
    DiagnosticCorrector,
    DiagnosticPhi,
    state_metrics,
)
from single_integrator.environment import Config, GiveWayEnv


class CLFHCBTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(corridor_half_length=1.3)
        self.goals = GiveWayEnv(self.config).goals

    def state(self, *, step=40, positions=None, velocity=None, history=None,
              candidate_since=None, terminal="running"):
        positions = np.asarray(
            [[-0.5, 0.0], [0.5, 0.0]] if positions is None else positions,
            dtype=np.float64,
        )
        velocity = np.asarray(
            [[0.1, 0.0], [-0.1, 0.0]] if velocity is None else velocity,
            dtype=np.float64,
        )
        if history is None:
            errors = np.linalg.norm(self.goals - positions, axis=-1)
            history = np.repeat(errors[None], min(step + 1, 41), axis=0)
        return AugmentedState(
            positions=positions,
            last_velocity=velocity,
            step=step,
            error_history=np.asarray(history, dtype=np.float64),
            history_start_step=step - len(history) + 1,
            candidate_since=candidate_since,
            terminal=terminal,
        )

    def test_corrector_is_deterministic_state_feedback_with_four_outputs(self):
        phi = DiagnosticPhi(0.2, -0.1, 0.15, "test")
        corrector = DiagnosticCorrector(phi)
        env = GiveWayEnv(self.config)
        safe = np.array([[0.2, 0.0], [-0.2, 0.0]])
        first = corrector(env.observation(), safe, self.config.max_speed)
        env.positions[0, 1] += 0.1
        second = corrector(env.observation(), safe, self.config.max_speed)
        self.assertEqual(first.shape, (2, 2))
        self.assertFalse(np.allclose(first, second))
        self.assertEqual(corrector.call_count, 2)
        self.assertFalse(corrector.stochastic)

    def test_timer_reset_low_progress_is_guarded_not_unresolved(self):
        state = self.state(candidate_since=None)
        metrics = state_metrics(state, self.config, self.goals)
        self.assertEqual(
            abstract_bin(metrics, self.config), BIN_INDEX["progress_starved"]
        )

    def test_resolved_progress_is_distinct_from_delayed_deadlock(self):
        positions = np.array([[0.35, 0.0], [-0.35, 0.0]])
        now = np.linalg.norm(self.goals - positions, axis=-1)
        history = np.repeat(now[None], 41, axis=0)
        history[0] = now + 0.02
        state = self.state(positions=positions, history=history)
        metrics = state_metrics(state, self.config, self.goals)
        self.assertEqual(
            abstract_bin(metrics, self.config), BIN_INDEX["resolved_progress"]
        )

    def test_empirical_bellman_table_has_terminal_boundaries(self):
        size = len(BIN_NAMES)
        p = np.zeros((size, 3 + size))
        # Every cell persists except resolved progress, which succeeds.
        for row in range(size):
            p[row, 3 + row] = 1.0
        p[BIN_INDEX["resolved_progress"]] = 0.0
        p[BIN_INDEX["resolved_progress"], 1] = 1.0
        table = np.zeros((4, size))
        for n in range(1, 4):
            table[n] = p[:, 0] + p[:, 3:] @ table[n - 1]
            table[n, :3] = 1.0
        cert = EmpiricalContinuationCertificate(
            phi=DiagnosticPhi(0, 0, 0, "zero"),
            horizon=3,
            transition_counts=np.asarray(p, dtype=np.int64),
            transition_probabilities=p,
            table=table,
        )
        self.assertGreaterEqual(cert.bellman_slacks().min(), -1e-12)
        for terminal, expected in [
            ("deadlock", 1.0),
            ("success", 0.0),
            ("collision", 0.0),
            ("timeout", 0.0),
        ]:
            metrics = state_metrics(self.state(terminal=terminal), self.config, self.goals)
            self.assertEqual(cert.value(metrics, 3, self.config), expected)
        running = state_metrics(self.state(), self.config, self.goals)
        self.assertEqual(cert.value(running, 0, self.config), 0.0)
        self.assertEqual(cert.value(running, 3, self.config), 1.0)

    def test_prefix_score_first_event_semantics(self):
        size = len(BIN_NAMES)
        p = np.zeros((size, 3 + size))
        p[:, 1] = 1.0
        table = np.zeros((self.config.max_steps + 1, size))
        table[1:, :3] = 1.0
        phi = DiagnosticPhi(0, 0, 0, "zero")
        cert = EmpiricalContinuationCertificate(
            phi=phi,
            horizon=self.config.max_steps,
            transition_counts=np.asarray(p, dtype=np.int64),
            transition_probabilities=p,
            table=table,
        )
        before = self.state(step=100)
        zeros = np.zeros((2, 2))
        for event, expected in [
            ("deadlock", 1.0),
            ("success", 0.0),
            ("collision", 0.0),
            ("timeout", 0.0),
        ]:
            after = self.state(step=101, terminal=event)
            step = ClosedLoopStep(
                state=before,
                observation=np.zeros((2, 10)),
                flow_key_data=np.zeros(2, dtype=np.uint32),
                u_flow=zeros,
                u_safe=zeros,
                correction=zeros,
                candidate=zeros,
                u_exec=zeros,
                first_projection_status="test",
                second_projection_status="test",
                first_min_cbf_residual=0.0,
                second_min_cbf_residual=0.0,
                next_state=after,
                event=event,
                monitor={},
            )
            trace = ClosedLoopTrace(
                rollout_id=f"test-{event}",
                pair_id=-1,
                flow_seed=0,
                phi=phi,
                initial_state=before,
                steps=(step,),
                outcome=event,
            )
            result = score_prefix(trace, 0, 20, cert, self.config)
            self.assertEqual(result.value, expected)
            self.assertEqual(result.prefix_event, event)


if __name__ == "__main__":
    unittest.main()
