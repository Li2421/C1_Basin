"""CPU-only source-freeze and generic-state collection contract tests."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent


def load(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceFreezeTests(unittest.TestCase):
    def test_authoritative_wide_generator_replays_frozen_asset(self) -> None:
        prepare = load("prepare_development_sources.py")
        helper = prepare.load_authoritative_helper()
        with np.load(prepare.REFERENCE_ASSET, allow_pickle=False) as payload:
            historical = np.asarray(payload["test_initial_positions"], dtype=np.float32)
        self.assertTrue(np.array_equal(helper.generate(2026090902, 200), historical))

    def test_two_deterministic_distinct_uniform_candidates(self) -> None:
        prepare = load("prepare_development_sources.py")
        first = prepare.anchor_steps("recovery_entry_id_dev_v1_0042")
        self.assertEqual(first, prepare.anchor_steps("recovery_entry_id_dev_v1_0042"))
        self.assertEqual(len(first), 2)
        self.assertEqual(len(set(first)), 2)
        self.assertEqual(first, sorted(first))
        self.assertTrue(all(0 <= value < 850 for value in first))
        draws = [
            value
            for source_index in range(120)
            for value in prepare.anchor_steps(f"recovery_entry_id_dev_v1_{source_index:04d}")
        ]
        # A gross regression to a fixed cadence or one terminal-relative window
        # would fail this broad-support check.  It is not a distributional test.
        self.assertLess(min(draws), 85)
        self.assertGreater(max(draws), 764)
        self.assertGreater(len(set(draws)), 180)

    def test_source_records_are_development_only_and_outcome_free(self) -> None:
        prepare = load("prepare_development_sources.py")
        initials = np.zeros((120, 2, 2), dtype=np.float32)
        sources, anchors = prepare.build_source_records(initials)
        self.assertEqual(len(sources), 120)
        self.assertEqual(len(anchors), 240)
        self.assertEqual(len({row["source_id"] for row in sources}), 120)
        self.assertTrue(all(row["split"] == "development" and row["development_only"] for row in sources))
        self.assertTrue(all(len(row["requested_anchor_steps"]) == 2 for row in sources))
        encoded = json.dumps({"sources": sources, "anchors": anchors}).lower()
        for forbidden in (
            "time_to_failure", "future_failure_type", "terminal_relative", "deadlock_onset",
            "recovery_success", "targeted_sampling",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_collector_accepts_only_exact_development_contract(self) -> None:
        collector = load("collect_safety_sources.py")
        rows = [{
            "source_id": f"root_{index:04d}", "split": "development", "development_only": True,
            "requested_anchor_steps": [index % 849, (index % 849) + 1],
        } for index in range(120)]
        source = {
            "development_only": True, "final_test_generated_or_used": False,
            "sources": {"development": rows},
        }
        self.assertEqual(collector.rows_for_collection(source), rows)
        bad = dict(source)
        bad["sources"] = {"development": rows, "final_test": []}
        with self.assertRaisesRegex(RuntimeError, "final-test leakage"):
            collector.rows_for_collection(bad)
        duplicate_anchor = json.loads(json.dumps(source))
        duplicate_anchor["sources"]["development"][0]["requested_anchor_steps"] = [4, 4]
        with self.assertRaisesRegex(RuntimeError, "two distinct"):
            collector.rows_for_collection(duplicate_anchor)

    def test_cpu_thread_cap_never_exceeds_three_per_allowed_shard(self) -> None:
        collector = load("collect_safety_sources.py")
        saved = {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS"
        )}
        try:
            os.environ["OMP_NUM_THREADS"] = "64"
            os.environ["OPENBLAS_NUM_THREADS"] = "2"
            os.environ.pop("MKL_NUM_THREADS", None)
            os.environ.pop("XLA_FLAGS", None)
            collector.cap_process_threads(3)
            self.assertEqual(int(os.environ["OMP_NUM_THREADS"]), 3)
            self.assertEqual(int(os.environ["OPENBLAS_NUM_THREADS"]), 2)
            self.assertEqual(int(os.environ["MKL_NUM_THREADS"]), 3)
            self.assertIn("intra_op_parallelism_threads=3", os.environ["XLA_FLAGS"])
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_protocol_matches_frozen_global_budget(self) -> None:
        protocol = json.loads((HERE / "collection_protocol.json").read_text())
        budget = json.loads((HERE / "experiment_budget.json").read_text())
        self.assertEqual(protocol["development_root_count"], budget["development_root_sources"])
        self.assertEqual(protocol["development_root_count"], 120)
        self.assertEqual(protocol["generic_anchor_rule"]["requested_anchors_per_root"], 2)
        self.assertEqual(budget["requested_states_per_root"], 2)
        self.assertEqual(protocol["maximum_new_branch_continuations"], 12000)
        self.assertEqual(protocol["maximum_new_branch_continuations"], budget["maximum_new_branch_continuations"])
        self.assertEqual(protocol["maximum_cpu_threads_total"], budget["cpu_threads_max"])
        self.assertEqual(protocol["maximum_cpu_threads_total"], 6)
        self.assertFalse(protocol["final_test_generated"])
        self.assertFalse(budget["final_test_used"])

    def test_no_outputs_exist_before_explicit_launch(self) -> None:
        # Tests/imports never freeze roots or launch Safety.  Once the explicit
        # commands run, this assertion deliberately becomes inapplicable.
        manifest = HERE / "development_source_manifest.json"
        if manifest.exists():
            return
        self.assertFalse((HERE / "runs").exists())
        self.assertFalse((HERE / "queried_state_manifest.json").exists())


class CompleteStateTests(unittest.TestCase):
    def test_finalizer_loads_exact_complete_nonterminal_state(self) -> None:
        finalizer = load("finalize_source_collection.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.npz"
            decision_path = root / "decision.npz"
            np.savez_compressed(
                state_path,
                positions=np.zeros((2, 2)), velocities=np.zeros((2, 2)), step=np.asarray(3),
                error_history=np.zeros((4, 2)), history_start_step=np.asarray(0),
                candidate_since=np.asarray(-1), stuck_timer=np.asarray(0), max_stuck_timer=np.asarray(0),
                ever_candidate_deadlock=np.asarray(False), first_success_step=np.asarray(-1),
                first_deadlock_step=np.asarray(-1), first_wall_collision_step=np.asarray(-1),
                first_agent_collision_step=np.asarray(-1), done=np.asarray(False),
                complete_real_history=np.asarray(True),
            )
            np.savez_compressed(
                decision_path, feature=np.zeros(214), u_flow=np.zeros((2, 2)),
                u_safe=np.zeros((2, 2)), flow_step_key=np.asarray([7, 9], dtype=np.uint32),
                observation=np.zeros(8), requested_global_step=np.asarray(3),
            )
            anchor = {
                "state_id": "s", "root_source_id": "r", "actual_global_step": 3,
                "state_file": str(state_path), "state_sha256": finalizer.sha256(state_path),
                "decision_input_file": str(decision_path),
                "decision_input_sha256": finalizer.sha256(decision_path),
                "flow_root_seed": 11, "flow_rollout_id": 12,
            }
            state = finalizer.load_materialized_state(anchor)
            self.assertEqual(len(state["h_t"]), 214)
            self.assertEqual(state["absolute_step"], 3)
            self.assertEqual(state["current_flow_key_data"], [7, 9])
            self.assertTrue(state["selection_terminal_relative_independent"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
