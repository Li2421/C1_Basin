"""Focused tests for the scenario-local four-agent expert dataset pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.expert import CoordinationHypothesis, ExpertPlan
from double_bottleneck.expert_dataset import (
    InitialConditionSpec,
    all_mode_rollout_requests,
    analyze_mode_diversity,
    assign_grouped_split,
    deduplicate_records,
    environment_descriptor,
    environment_fingerprint,
    infer_coordination_mode,
    initialize_environment,
    load_record,
    pilot_initial_condition_specs,
    pilot_rollout_requests,
    record_from_plan,
    save_dataset,
    trajectory_digest,
    validate_record,
    validate_records,
)
from double_bottleneck.generate_expert_dataset import main as generation_main
from double_bottleneck.generate_expert_dataset_8mode import main as eight_mode_generation_main


def _one_step_success_record(rollout_id: str = "unit"):
    base = DoubleBottleneckEnv()
    velocities = np.asarray(
        ((0.03, 0.01), (0.02, -0.01), (-0.03, -0.01), (-0.02, 0.01)),
        dtype=np.float64,
    )
    spec = InitialConditionSpec(
        condition_id="goals",
        family_id="goals_family",
        regime="weakly_asymmetric",
        positions=base.goals.copy(),
        initial_velocities=velocities,
        perturbation={"test": True},
    )
    source = initialize_environment(Config(), spec)
    validation = DoubleBottleneckEnv(source.config)
    validation.restore_augmented_state(source.augmented_state())
    positions = [validation.positions.copy()]
    observations = [validation.observation().copy()]
    action = np.zeros((4, 2), dtype=np.float64)
    observation, _, done, info = validation.step(action)
    assert done and info["termination"] == "success"
    positions.append(validation.positions.copy())
    observations.append(observation.copy())
    plan = ExpertPlan(
        hypothesis=CoordinationHypothesis("left_to_right"),
        positions=np.asarray(positions),
        actions=action[None, ...],
        observations=np.asarray(observations),
        terminal_reason="success",
        success=True,
        collision=False,
        timeout=False,
        deadlock=False,
        episode_steps=1,
        solve_time_seconds=0.001,
        path_length=0.0,
        min_inter_agent_surface_distance=float(info["min_swept_agent_distance"]),
        min_wall_clearance=float(info["min_swept_wall_distance"]),
    )
    return record_from_plan(plan, source, spec, rollout_id)


class PilotDesignTests(unittest.TestCase):
    def test_restrained_design_has_twelve_paired_initial_states(self):
        specs = pilot_initial_condition_specs()
        requests = pilot_rollout_requests(specs)
        self.assertEqual(len(specs), 12)
        self.assertEqual(len(requests), 24)
        for regime in ("clearly_asymmetric", "weakly_asymmetric", "near_symmetric"):
            self.assertEqual(sum(spec.regime == regime for spec in specs), 4)
        self.assertEqual(sum(np.any(spec.initial_velocities) for spec in specs), 3)
        for spec in specs:
            paired = [hypothesis for candidate, hypothesis in requests if candidate is spec]
            self.assertEqual(
                {hypothesis.first_direction for hypothesis in paired},
                {"left_to_right", "right_to_left"},
            )
            env = initialize_environment(Config(), spec)
            np.testing.assert_array_equal(env.positions, spec.positions)
            np.testing.assert_array_equal(env.velocities, spec.initial_velocities)

    def test_all_mode_design_has_eight_unique_hypotheses_per_start(self):
        specs = pilot_initial_condition_specs()
        requests = all_mode_rollout_requests(specs)
        self.assertEqual(len(requests), 96)
        for spec in specs:
            hypotheses = [hypothesis for candidate, hypothesis in requests if candidate is spec]
            self.assertEqual(len(hypotheses), 8)
            self.assertEqual(len({hypothesis.label for hypothesis in hypotheses}), 8)

    def test_environment_fingerprint_includes_geometry_and_semantics(self):
        env = DoubleBottleneckEnv()
        descriptor = environment_descriptor(env)
        self.assertEqual(environment_fingerprint(env), environment_fingerprint(descriptor))
        changed = dict(descriptor)
        changed["observation_schema"] = "changed"
        self.assertNotEqual(environment_fingerprint(env), environment_fingerprint(changed))


class RecordValidationTests(unittest.TestCase):
    def test_record_replay_preserves_nonzero_initial_velocity_observation(self):
        record = _one_step_success_record()
        validate_record(record)
        self.assertEqual(record.positions.shape, (2, 4, 2))
        self.assertEqual(record.observations.shape, (2, 4, 18))
        self.assertEqual(record.actions.shape, (1, 4, 2))
        np.testing.assert_allclose(
            record.observations[0, :, 2:4], record.initial_velocities, rtol=0.0, atol=2e-9
        )
        self.assertTrue(record.training_eligible)

    def test_corrupt_dynamics_and_success_labels_are_rejected(self):
        record = _one_step_success_record()
        positions = record.positions.copy()
        positions[-1, 0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "dynamics"):
            validate_record(replace(record, positions=positions))
        with self.assertRaisesRegex(ValueError, "label"):
            validate_record(replace(record, success=False))

    def test_exact_duplicate_detection(self):
        first = _one_step_success_record("first")
        second = replace(first, rollout_id="second")
        unique, duplicates = deduplicate_records((first, second))
        self.assertEqual([record.rollout_id for record in unique], ["first"])
        self.assertEqual(duplicates[0]["duplicate_of"], "first")
        with self.assertRaisesRegex(ValueError, "duplicate trajectory"):
            validate_records((first, second))

    def test_round_trip_uses_non_object_npz_and_revalidates(self):
        record = replace(_one_step_success_record(), split="train")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            manifest = save_dataset(output, (record,))
            loaded = load_record(output / manifest["files"][0]["file"])
            validate_record(loaded)
            np.testing.assert_array_equal(loaded.observations, record.observations)
            np.testing.assert_array_equal(loaded.actions, record.actions)

    def test_grouped_split_never_separates_paired_modes(self):
        base = _one_step_success_record()
        records = []
        for regime in ("clearly_asymmetric", "weakly_asymmetric", "near_symmetric"):
            for family_index in range(2):
                family = f"{regime}_{family_index}"
                for mode_index in range(2):
                    records.append(
                        replace(
                            base,
                            rollout_id=f"{family}_{mode_index}",
                            family_id=family,
                            regime=regime,
                        )
                    )
        assigned = assign_grouped_split(records, val_fraction=0.5, seed=3)
        by_family = {}
        for record in assigned:
            by_family.setdefault(record.family_id, set()).add(record.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_family.values()))
        for regime in ("clearly_asymmetric", "weakly_asymmetric", "near_symmetric"):
            self.assertEqual(
                {record.split for record in assigned if record.regime == regime}, {"train", "val"}
            )


class CoordinationModeTests(unittest.TestCase):
    @staticmethod
    def _crossing_positions(ltr_first: bool) -> np.ndarray:
        time = np.arange(41, dtype=np.float64)
        delays = (0, 2, 12, 14) if ltr_first else (12, 14, 0, 2)
        values = np.empty((len(time), 4, 2), dtype=np.float64)
        for agent in range(4):
            direction = 1.0 if agent < 2 else -1.0
            start = -3.0 if direction > 0 else 3.0
            finish = -start
            alpha = np.clip((time - delays[agent]) / 20.0, 0.0, 1.0)
            values[:, agent, 0] = start + alpha * (finish - start)
            values[:, agent, 1] = (-0.3, 0.3, 0.3, -0.3)[agent]
        return values

    def test_modes_come_from_actual_crossings_not_hypothesis_labels(self):
        env = DoubleBottleneckEnv()
        ltr = self._crossing_positions(True)
        rtl = self._crossing_positions(False)
        ltr_mode = infer_coordination_mode(ltr, env.goals, env.config)
        rtl_mode = infer_coordination_mode(rtl, env.goals, env.config)
        self.assertTrue(ltr_mode["complete"] and rtl_mode["complete"])
        self.assertEqual(ltr_mode["first_direction"], "left_to_right")
        self.assertEqual(rtl_mode["first_direction"], "right_to_left")
        self.assertNotEqual(ltr_mode["signature"], rtl_mode["signature"])

        base = _one_step_success_record()
        common_start = ltr[0].copy()
        records = []
        for index, (positions, mode) in enumerate(((ltr, ltr_mode), (rtl, rtl_mode))):
            records.append(
                replace(
                    base,
                    rollout_id=f"mode_{index}",
                    initial_positions=common_start,
                    positions=positions,
                    coordination_mode=mode,
                    trajectory_digest=trajectory_digest(positions, base.actions),
                )
            )
        analysis = analyze_mode_diversity(records)
        self.assertTrue(analysis["genuine_multimodality_observed"])
        self.assertEqual(analysis["identical_initial_states_with_multiple_successful_modes"], 1)


class GenerationGateTests(unittest.TestCase):
    def test_cli_cannot_solve_or_write_without_gate_a_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "forbidden"
            with self.assertRaisesRegex(SystemExit, "Gate A"):
                generation_main(("--output", str(output)))
            self.assertFalse(output.exists())

    def test_eight_mode_cli_cannot_solve_or_write_without_gate_a_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "forbidden_eight_mode"
            with self.assertRaisesRegex(SystemExit, "Gate A"):
                eight_mode_generation_main(("--output", str(output)))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
