"""CPU-only integrity tests for the single-segment recovery state machine."""

from __future__ import annotations

import hashlib
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/single_segment_recovery_training_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
for value in (str(SYSROOT), str(ROOT), str(PILOT), str(HERE)):
    if value in sys.path:
        sys.path.remove(value)
sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT), str(HERE)]

from state_machine import (  # noqa: E402
    AuthoritativeStepKernel,
    ControllerMemory,
    Mode,
    RecoverySystem,
    SingleSegmentRecoveryMachine,
    TerminalStateError,
)


@dataclass
class FakeConfig:
    max_steps: int = 20
    max_speed: float = 1.0
    dt: float = 0.05


@dataclass
class FakeCBF:
    feasibility_tol: float = 1e-8
    speed_tol: float = 1e-8


class FakeEnv:
    def __init__(self, *, step: int = 0, max_steps: int = 20):
        self.step_count = step
        self.max_steps = max_steps
        self.done = False
        self.positions = np.array([[-0.8, 0.0], [0.8, 0.0]], dtype=np.float64)
        self.velocities = np.zeros((2, 2), dtype=np.float64)
        self.distance_history = [np.array([1.0, 1.0], dtype=np.float64)]
        self.executed: list[np.ndarray] = []

    def observation(self):
        value = np.zeros((2, 10), dtype=np.float32)
        value[:, 4] = np.array([1.0, -1.0])
        value[:, 6] = np.array([1.0, -1.0])
        return value

    def snapshot(self):
        return {"step": self.step_count}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.executed.append(action.copy())
        self.velocities = action.copy()
        self.positions += 0.05 * action
        self.step_count += 1
        self.distance_history.append(np.array([1.0, 1.0]))
        self.done = self.step_count >= self.max_steps
        event = "timeout" if self.done else None
        return self.observation(), 0.0, self.done, {"termination": event}


class FakeBuilder:
    def __init__(self):
        self.calls = 0

    def build(self, env, first, config, cbf):
        self.calls += 1
        assert np.isfinite(np.asarray(env.distance_history)).all()
        feature = np.full(214, env.step_count, dtype=np.float64)
        feature[:4] = np.asarray(first["u_safe"]).reshape(4)
        return feature, {}


class CountingFlow:
    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, observation, key):
        self.calls.append(int(key))
        return np.full((2, 2), 0.2, dtype=np.float64)


class FakeDirect:
    def __init__(self):
        self.calls = 0

    def __call__(self, feature_batch):
        self.calls += 1
        return np.full((1, 4), 0.1 * self.calls, dtype=np.float64)


class FakeEta:
    def __init__(self):
        self.calls = 0

    def predict(self, feature_batch):
        self.calls += 1
        eta = np.array([[0.25, -0.5, 0.75]], dtype=np.float64)
        return eta, eta.copy(), eta.copy()


class CountingCorrector:
    def __init__(self, eta):
        self.eta = np.asarray(eta, dtype=np.float64).copy()
        self.calls = 0

    def __call__(self, observation, u_safe, max_speed):
        self.calls += 1
        return np.full((2, 2), self.eta[0], dtype=np.float64)


def identity_project(action, matrix, lower, max_speed, cbf):
    return np.asarray(action, dtype=np.float64), "ok", False, {}


def fake_constraints(snapshot, cbf):
    return np.zeros((1, 4)), np.zeros(1), {}


def make_kernel(env: FakeEnv, flow: CountingFlow | None = None):
    flow = flow or CountingFlow()
    kernel = AuthoritativeStepKernel(
        env=env,
        config=FakeConfig(max_steps=env.max_steps),
        cbf=FakeCBF(),
        episode_key=0,
        sample_action=flow,
        feature_builder=FakeBuilder(),
        project=identity_project,
        key_for_step=lambda episode_key, step: step,
        bounded_action=lambda action, max_speed: np.asarray(action),
        constraints=fake_constraints,
    )
    return kernel, flow


class StateMachineUnitTests(unittest.TestCase):
    def test_enter_executes_recovery_same_step_and_flow_is_sampled_once(self):
        env = FakeEnv()
        kernel, flow = make_kernel(env)
        direct = FakeDirect()
        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: True, exit_head=lambda x: True,
            direct_model=direct,
        )
        record = machine.step()
        self.assertEqual(record.decision, "ENTER")
        self.assertEqual(record.mode_after_decision, Mode.RECOVERY)
        self.assertEqual(record.entry_step, 0)
        self.assertEqual(record.recovery_transitions, 1)
        self.assertFalse(record.exit_queried)
        self.assertEqual(record.flow_sample_count, 1)
        self.assertEqual(flow.calls, [0])
        self.assertEqual(direct.calls, 1)
        np.testing.assert_allclose(record.u_exec, 0.3)

    def test_exit_next_step_executes_safety_and_prevents_reentry(self):
        env = FakeEnv(max_steps=5)
        kernel, flow = make_kernel(env)
        direct = FakeDirect()
        entry_calls = []
        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: entry_calls.append(True) or True,
            exit_head=lambda x: True,
            direct_model=direct,
        )
        first = machine.step()
        second = machine.step()
        third = machine.step()
        self.assertEqual([first.decision, second.decision, third.decision],
                         ["ENTER", "EXIT", "SAFETY_AFTER"])
        self.assertEqual(second.exit_step, 1)
        self.assertEqual(direct.calls, 1)
        self.assertEqual(len(entry_calls), 1)
        self.assertEqual(flow.calls, [0, 1, 2])
        np.testing.assert_allclose(second.u_exec, second.u_safe)
        np.testing.assert_allclose(third.u_exec, third.u_safe)

    def test_direct_model_is_requeried_on_every_active_transition(self):
        env = FakeEnv(max_steps=4)
        kernel, _ = make_kernel(env)
        direct = FakeDirect()
        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: True, exit_head=lambda x: False,
            direct_model=direct,
        )
        records = [machine.step() for _ in range(3)]
        self.assertEqual(direct.calls, 3)
        self.assertEqual(machine.direct_query_count, 3)
        self.assertTrue(all(row.recovery_action for row in records))
        self.assertEqual([row.recovery_transitions for row in records], [1, 2, 3])

    def test_second_projection_occurs_on_every_recovery_action_only(self):
        env = FakeEnv(max_steps=6)
        calls = []

        def project(action, matrix, lower, max_speed, cbf):
            calls.append(np.asarray(action).copy())
            return np.asarray(action, dtype=np.float64), "ok", False, {}

        flow = CountingFlow()
        kernel = AuthoritativeStepKernel(
            env=env, config=FakeConfig(max_steps=env.max_steps), cbf=FakeCBF(),
            episode_key=0, sample_action=flow, feature_builder=FakeBuilder(),
            project=project, key_for_step=lambda episode_key, step: step,
            bounded_action=lambda action, max_speed: np.asarray(action),
            constraints=fake_constraints,
        )
        decisions = iter((False, True))
        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: True, exit_head=lambda x: next(decisions),
            direct_model=FakeDirect(),
        )
        rows = [machine.step(), machine.step(), machine.step()]
        self.assertEqual([row.decision for row in rows], ["ENTER", "CONTINUE", "EXIT"])
        # Three first projections plus exactly two recovery second projections.
        self.assertEqual(len(calls), 5)
        self.assertEqual(sum(row.recovery_action for row in rows), 2)
        self.assertEqual(machine.direct_query_count, 2)

    def test_eta_is_predicted_once_frozen_and_exit_gets_217_dimensions(self):
        env = FakeEnv(max_steps=5)
        kernel, _ = make_kernel(env)
        eta = FakeEta()
        correctors: list[CountingCorrector] = []
        exit_shapes = []

        def factory(value):
            correctors.append(CountingCorrector(value))
            return correctors[-1]

        def exit_head(value):
            exit_shapes.append((value.shape, value[-3:].copy()))
            return len(exit_shapes) == 2

        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.STRUCTURED_ETA,
            entry_head=lambda x: True, exit_head=exit_head,
            eta_model=eta, eta_corrector_factory=factory,
        )
        rows = [machine.step(), machine.step(), machine.step()]
        self.assertEqual(eta.calls, 1)
        self.assertEqual(machine.eta_query_count, 1)
        self.assertEqual(correctors[0].calls, 2)
        self.assertEqual([row.decision for row in rows], ["ENTER", "CONTINUE", "EXIT"])
        self.assertEqual([shape for shape, _ in exit_shapes], [(217,), (217,)])
        for _, value in exit_shapes:
            np.testing.assert_array_equal(value, [0.25, -0.5, 0.75])
        np.testing.assert_array_equal(machine.memory.eta_latched, [0.25, -0.5, 0.75])
        self.assertFalse(machine.memory.eta_latched.flags.writeable)

    def test_recovery_branch_restores_eta_without_reprediction(self):
        env = FakeEnv(step=12, max_steps=15)
        kernel, _ = make_kernel(env)
        eta = FakeEta()
        correctors = []

        def factory(value):
            correctors.append(CountingCorrector(value))
            return correctors[-1]

        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.STRUCTURED_ETA,
            entry_head=lambda x: False, exit_head=lambda x: False,
            eta_model=eta, eta_corrector_factory=factory,
        )
        machine.restore_controller_memory(ControllerMemory(
            mode=Mode.RECOVERY, eta_latched=np.array([0.4, -0.2, 0.1]),
            recovery_used=True, entry_step=5, recovery_transitions=7,
        ))
        row = machine.step()
        self.assertEqual(row.decision, "CONTINUE")
        self.assertEqual(row.recovery_transitions, 8)
        self.assertEqual(eta.calls, 0)
        self.assertEqual(machine.eta_query_count, 0)
        self.assertEqual(correctors[0].calls, 1)
        np.testing.assert_array_equal(row.eta_latched, [0.4, -0.2, 0.1])

    def test_terminal_state_is_absorbing_and_consumes_no_flow(self):
        env = FakeEnv()
        env.done = True
        kernel, flow = make_kernel(env)
        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: True, exit_head=lambda x: False,
            direct_model=FakeDirect(),
        )
        with self.assertRaises(TerminalStateError):
            machine.step()
        self.assertEqual(flow.calls, [])
        self.assertEqual(env.executed, [])

    def test_absolute_step_is_not_reset_on_takeover(self):
        env = FakeEnv(step=13, max_steps=18)
        kernel, flow = make_kernel(env)
        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: True, exit_head=lambda x: True,
            direct_model=FakeDirect(),
        )
        first, second = machine.step(), machine.step()
        self.assertEqual(first.entry_step, 13)
        self.assertEqual(second.exit_step, 14)
        self.assertEqual(flow.calls, [13, 14])
        self.assertEqual(env.step_count, 15)

    def test_feature_padding_view_does_not_mutate_monitor_history(self):
        env = FakeEnv(step=3)
        env.distance_history = [np.full(2, np.nan), np.full(2, np.nan),
                                np.array([2.0, 2.0]), np.array([1.0, 1.0])]
        original = [value.copy() for value in env.distance_history]
        kernel, _ = make_kernel(env)
        machine = SingleSegmentRecoveryMachine(
            kernel=kernel, system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: False, exit_head=lambda x: False,
            direct_model=FakeDirect(),
        )
        machine.step()
        for observed, expected in zip(env.distance_history[:4], original):
            np.testing.assert_equal(observed, expected)


class FrozenControllerSmokeTests(unittest.TestCase):
    DIRECT = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz"
    ETA = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
    EXPECTED_DIRECT = "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700"
    EXPECTED_ETA = "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095"

    @staticmethod
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def setUp(self):
        self.assertEqual(self.sha(self.DIRECT), self.EXPECTED_DIRECT)
        self.assertEqual(self.sha(self.ETA), self.EXPECTED_ETA)

    @staticmethod
    def make_real_kernel(env, config, cbf):
        from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
        from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry

        return AuthoritativeStepKernel(
            env=env, config=config, cbf=cbf, episode_key=0,
            sample_action=lambda observation, key: np.array([[0.35, 0.02], [-0.35, -0.02]]),
            feature_builder=StartupAwareFeatureBuilder(), project=project_velocity_with_retry,
            key_for_step=lambda episode_key, step: np.asarray([0, step], dtype=np.uint32),
        )

    @staticmethod
    def manual_rollout(env, config, cbf, *, direct=None, eta=None, steps=3):
        from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
        from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
        from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
        from single_integrator.cbf import barrier_constraints
        from single_integrator.environment import bounded_nominal

        builder = StartupAwareFeatureBuilder()
        corrector = None
        records = []
        for _ in range(steps):
            observation = np.asarray(env.observation(), dtype=np.float32)
            raw = np.array([[0.35, 0.02], [-0.35, -0.02]], dtype=np.float64)
            u_flow = bounded_nominal(raw, config.max_speed)
            matrix, lower, _ = barrier_constraints(env.snapshot(), cbf)
            u_safe, _, retry1, _ = project_velocity_with_retry(
                u_flow, matrix, lower, config.max_speed, cbf
            )
            feature, _ = builder.build(
                env, {"u_flow": u_flow, "u_safe": u_safe}, config, cbf
            )
            if direct is not None:
                correction = direct(feature[None])[0].reshape(2, 2)
            else:
                if corrector is None:
                    predicted, _, _ = eta.predict(feature[None])
                    corrector = DiagnosticCorrector(DiagnosticPhi(*predicted[0].tolist()))
                correction = corrector(observation.astype(np.float64), u_safe, config.max_speed)
            u_exec, _, retry2, _ = project_velocity_with_retry(
                u_safe + correction, matrix, lower, config.max_speed, cbf
            )
            env.step(u_exec)
            records.append((u_flow.copy(), u_safe.copy(), correction.copy(), u_exec.copy(), bool(retry1), bool(retry2)))
        return records

    @staticmethod
    def manual_safety(env, config, cbf):
        from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
        from single_integrator.cbf import barrier_constraints
        from single_integrator.environment import bounded_nominal

        actions = []
        while not env.done:
            raw = np.array([[0.35, 0.02], [-0.35, -0.02]], dtype=np.float64)
            u_flow = bounded_nominal(raw, config.max_speed)
            matrix, lower, _ = barrier_constraints(env.snapshot(), cbf)
            u_safe, _, _, _ = project_velocity_with_retry(u_flow, matrix, lower, config.max_speed, cbf)
            env.step(u_safe)
            actions.append(u_safe.copy())
        return actions

    def test_real_direct_and_eta_controller_contracts_cpu(self):
        from pilot_common import DeterministicGphi
        from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor
        from single_integrator.cbf import CBFConfig
        from single_integrator.environment import Config, GiveWayEnv

        config = Config(max_steps=6)
        cbf = CBFConfig()
        initial = np.array([[-0.9, 0.01], [0.9, -0.01]], dtype=np.float64)
        direct = DeterministicGphi(self.DIRECT)
        eta = FixedDEtaPredictor(self.ETA)

        env_g = GiveWayEnv(config); env_g.reset(initial)
        ref_g = GiveWayEnv(config); ref_g.reset(initial)
        g_machine = SingleSegmentRecoveryMachine(
            kernel=self.make_real_kernel(env_g, config, cbf),
            system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: True, exit_head=lambda x: False,
            direct_model=direct,
        )
        g_rows = [g_machine.step() for _ in range(3)]
        g_reference = self.manual_rollout(ref_g, config, cbf, direct=direct, steps=3)
        self.assertEqual(g_machine.direct_query_count, 3)
        self.assertEqual(g_machine.kernel.flow_sample_count, 3)
        self.assertEqual(env_g.step_count, 3)
        self.assertTrue(all(np.isfinite(row.u_exec).all() for row in g_rows))
        for row, expected in zip(g_rows, g_reference):
            for observed, reference in zip(
                (row.u_flow, row.u_safe, row.correction, row.u_exec), expected[:4]
            ):
                np.testing.assert_array_equal(observed, reference)
        np.testing.assert_array_equal(env_g.positions, ref_g.positions)
        np.testing.assert_array_equal(env_g.velocities, ref_g.velocities)
        np.testing.assert_array_equal(env_g.distance_history, ref_g.distance_history)

        env_eta = GiveWayEnv(config); env_eta.reset(initial)
        ref_eta = GiveWayEnv(config); ref_eta.reset(initial)
        eta_machine = SingleSegmentRecoveryMachine(
            kernel=self.make_real_kernel(env_eta, config, cbf),
            system=RecoverySystem.STRUCTURED_ETA,
            entry_head=lambda x: True, exit_head=lambda x: False,
            eta_model=eta,
        )
        eta_rows = [eta_machine.step() for _ in range(3)]
        eta_reference = self.manual_rollout(ref_eta, config, cbf, eta=eta, steps=3)
        self.assertEqual(eta_machine.eta_query_count, 1)
        self.assertEqual(eta_machine.kernel.flow_sample_count, 3)
        self.assertEqual(env_eta.step_count, 3)
        np.testing.assert_array_equal(eta_rows[0].eta_latched, eta_rows[-1].eta_latched)
        self.assertTrue(all(np.isfinite(row.u_exec).all() for row in eta_rows))
        for row, expected in zip(eta_rows, eta_reference):
            for observed, reference in zip(
                (row.u_flow, row.u_safe, row.correction, row.u_exec), expected[:4]
            ):
                np.testing.assert_array_equal(observed, reference)
        np.testing.assert_array_equal(env_eta.positions, ref_eta.positions)
        np.testing.assert_array_equal(env_eta.velocities, ref_eta.velocities)
        np.testing.assert_array_equal(env_eta.distance_history, ref_eta.distance_history)

    def test_safety_only_machine_is_bitwise_equal_to_manual_safety(self):
        from pilot_common import DeterministicGphi
        from single_integrator.cbf import CBFConfig
        from single_integrator.environment import Config, GiveWayEnv

        config = Config(max_steps=5)
        cbf = CBFConfig()
        initial = np.array([[-0.9, 0.01], [0.9, -0.01]], dtype=np.float64)
        env_machine = GiveWayEnv(config); env_machine.reset(initial)
        env_reference = GiveWayEnv(config); env_reference.reset(initial)
        machine = SingleSegmentRecoveryMachine(
            kernel=self.make_real_kernel(env_machine, config, cbf),
            system=RecoverySystem.DIRECT_G,
            entry_head=lambda x: False, exit_head=lambda x: False,
            direct_model=DeterministicGphi(self.DIRECT),
        )
        rows = machine.run_to_terminal()
        reference_actions = self.manual_safety(env_reference, config, cbf)
        self.assertEqual([row.decision for row in rows], ["WAIT"] * 5)
        for row, reference in zip(rows, reference_actions):
            np.testing.assert_array_equal(row.u_exec, reference)
        np.testing.assert_array_equal(env_machine.positions, env_reference.positions)
        np.testing.assert_array_equal(env_machine.velocities, env_reference.velocities)
        np.testing.assert_array_equal(env_machine.distance_history, env_reference.distance_history)
        self.assertEqual(env_machine.summary(), env_reference.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
