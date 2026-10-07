"""Regression tests for the global single-segment experiment budget ledger."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_paired_branch_manifest as branch_manifest


def _canonical_payload(body: dict) -> dict:
    payload = dict(body)
    payload["content_sha256"] = branch_manifest.canonical_hash(payload)
    return payload


class SourceBudgetTests(unittest.TestCase):
    def test_completed_eta_collector_budget_is_counted_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            (runs / "source_collection_runtime_shard0.json").write_text(json.dumps({
                "new_physical_steps": 123,
            }))
            audit = _canonical_payload({
                "schema": "single_segment_eta_recovery_state_collection_integrity_v1",
                "status": "PASS",
                "budget_use": {
                    "new_continuations": 24,
                    "new_physical_simulation_steps": 381,
                },
            })
            (root / "eta_recovery_state_collection_integrity.json").write_text(json.dumps(audit))
            # If both formats happen to exist, the compatibility artifact must
            # not double-count the same recovery-state collection.
            legacy = _canonical_payload({
                "collection_budget": {
                    "new_prefix_continuations": 999,
                    "new_physical_steps": 999,
                },
            })
            (root / "eta_recovery_decision_state_manifest.json").write_text(json.dumps(legacy))
            with mock.patch.object(branch_manifest, "HERE", root):
                continuations, steps, assets = branch_manifest.source_collection_budget()
            self.assertEqual(continuations, 24)
            self.assertEqual(steps, 504)
            self.assertEqual(len(assets), 2)
            self.assertTrue(assets[-1].endswith("eta_recovery_state_collection_integrity.json"))

    def test_nonpassing_eta_collector_audit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = _canonical_payload({
                "schema": "single_segment_eta_recovery_state_collection_integrity_v1",
                "status": "INCOMPLETE",
                "budget_use": {
                    "new_continuations": 1,
                    "new_physical_simulation_steps": 1,
                },
            })
            (root / "eta_recovery_state_collection_integrity.json").write_text(json.dumps(audit))
            with mock.patch.object(branch_manifest, "HERE", root):
                with self.assertRaises(RuntimeError):
                    branch_manifest.source_collection_budget()


if __name__ == "__main__":
    unittest.main()
