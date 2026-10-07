"""Tests for bounded, outcome-blind policy-iteration state collection."""

from __future__ import annotations

import unittest

from build_paired_branch_manifest import source_diverse_select
from collect_policy_iteration_states import select_bounded


class PolicyIterationSelectionTests(unittest.TestCase):
    def test_bounded_selection_is_source_diverse_and_rank_only(self) -> None:
        rows = []
        for source in range(15):
            rows.extend([
                {"root_source_id": f"s{source:02d}", "selection_rank": f"{source:02d}b", "outcome": "fail"},
                {"root_source_id": f"s{source:02d}", "selection_rank": f"{source:02d}a", "outcome": "success"},
            ])
        selected = select_bounded(rows, maximum=12)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row["root_source_id"] for row in selected}), 12)
        self.assertTrue(all(row["selection_rank"].endswith("a") for row in selected))

    def test_policy_iteration_manifest_selection_uses_all_train_quota(self) -> None:
        rows = [
            {"root_source_id": f"train_{index:02d}", "state_id": f"state_{index:02d}", "split": "train"}
            for index in range(15)
        ]
        selected = source_diverse_select(rows, "policy_iteration_entry", 12)
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(row["split"] == "train" for row in selected))


if __name__ == "__main__":
    unittest.main()
