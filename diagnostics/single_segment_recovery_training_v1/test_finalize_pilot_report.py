import tempfile
import unittest
import json
from pathlib import Path

import finalize_pilot_report as report


class FinalizePilotReportTests(unittest.TestCase):
    def test_incomplete_policy_iteration_is_explicitly_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "branch_manifests").mkdir()
            pi = report.build_policy_iteration_manifest(root)
            decision = report.classify(
                pi,
                {},
                False,
                [],
                {"within_budget": True},
            )
            self.assertTrue(pi["interim_partial"])
            self.assertEqual(decision["status"], "PARTIAL")
            self.assertIsNone(decision["classification"])

    def test_non_owned_json_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "integrity_checks.json"
            path.write_text('{"authoritative": true}\n')
            with self.assertRaises(RuntimeError):
                report.write_json(path, {"schema": "test"}, dry_run=False)

    def test_owned_json_can_be_refreshed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime_statistics.json"
            report.write_json(path, {"schema": "test", "value": 1}, dry_run=False)
            report.write_json(path, {"schema": "test", "value": 2}, dry_run=False)
            self.assertEqual(report.read_json(path)["value"], 2)

    def test_policy_iteration_bundle_is_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy_iteration_validation.json").write_text(json.dumps({
                "selection": {"selected_policy_hash": "f6", "selected_metrics": {"q_learned": 0.9}}
            }))
            (root / "policy_iteration_calibration.json").write_text(json.dumps({
                "metrics": {"q_learned": 0.8}, "break_budget_audit": {"break_budget": 0.05}
            }))
            (root / "final_test_results.json").write_text(json.dumps({
                "metrics": {"q_learned": 0.91}
            }))
            bundle = report.load_evaluation_report(root, root / "missing.json")
            self.assertEqual(bundle["selection"]["selected_policy_hash"], "f6")
            self.assertEqual(bundle["final_test"]["q_learned"], 0.91)

    def test_strong_final_test_supports_single_segment(self):
        final = {
            "source_episode_count": 200,
            "q_safe": 0.715,
            "q_learned": 0.905,
            "break": 0,
            "delta_q_paired_ci95": [0.11, 0.26],
            "hard_safety": {"intact": True},
            "segment_statistics": {"successful_finite_segment_then_safety_count": 181},
        }
        bundle = {
            "selection": {"selected_metrics": {"segment_statistics": {"successful_finite_segment_then_safety_count": 19}}},
            "calibration": {"delta_q": 0.15},
            "final_test": final,
        }
        decision = report.classify(
            {"interim_partial": False}, bundle, True,
            [{"root_source_id": str(i)} for i in range(200)],
            {"within_budget": True},
        )
        self.assertEqual(decision["classification"], "SINGLE_SEGMENT_RECOVERY_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
