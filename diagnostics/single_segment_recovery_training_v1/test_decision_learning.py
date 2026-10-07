"""CPU-only unit tests for decision_learning.py."""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from decision_learning import (
    DecisionEvidence,
    DeformationNormalizer,
    MatchedBranchPair,
    decision_loss_components,
    fit_feature_normalization,
    head_logits,
    init_head_parameters,
    pair_branch_records,
    paired_success_difference_interval,
    predict_head_logit,
    predict_head_probability,
    save_decision_head_checkpoint,
    summarize_decision_evidence,
    train_decision_head,
)


def pairs(cells: list[tuple[int, int]], j0: float = 2.0, j1: float = 1.0):
    return [
        MatchedBranchPair(str(index), success0, success1, j0, j1)
        for index, (success0, success1) in enumerate(cells)
    ]


class PairingTests(unittest.TestCase):
    def test_strict_pairing_and_cell_counts(self):
        rows = []
        cells = [(0, 0), (0, 1), (0, 1), (1, 0), (1, 1)]
        for stream, (success0, success1) in enumerate(cells):
            common = {
                "decision_id": "d",
                "future_stream_id": f"f{stream}",
                "state_hash": "same",
                "current_flow_realization_id": "flow",
                "remaining_jdef": 0.1,
            }
            rows.extend(
                [
                    {**common, "action": 0, "success": success0},
                    {**common, "action": 1, "success": success1},
                ]
            )
        paired = pair_branch_records(reversed(rows))["d"]
        evidence = summarize_decision_evidence("d", paired, minimum_pairs=1)
        self.assertEqual((evidence.both_fail, evidence.rescue, evidence.break_count, evidence.both_success), (1, 2, 1, 1))
        self.assertAlmostEqual(evidence.r, 0.4)
        self.assertAlmostEqual(evidence.b, 0.2)
        self.assertAlmostEqual(evidence.paired_delta, 0.2)

    def test_missing_or_mismatched_pair_raises(self):
        with self.assertRaisesRegex(ValueError, "missing matched"):
            pair_branch_records(
                [{"decision_id": "d", "future_stream_id": "f", "action": 0, "success": 1, "remaining_jdef": 0.0}]
            )
        rows = [
            {"decision_id": "d", "future_stream_id": "f", "action": 0, "success": 1, "remaining_jdef": 0.0, "state_hash": "a"},
            {"decision_id": "d", "future_stream_id": "f", "action": 1, "success": 1, "remaining_jdef": 0.0, "state_hash": "b"},
        ]
        with self.assertRaisesRegex(ValueError, "state_hash"):
            pair_branch_records(rows)


class UncertaintyTests(unittest.TestCase):
    def test_no_discordance_at_32_is_not_equivalence(self):
        evidence = summarize_decision_evidence("d", pairs([(1, 1)] * 32))
        self.assertFalse(evidence.equivalence_supported)
        self.assertTrue(evidence.unresolved)
        self.assertEqual(evidence.status, "UNRESOLVED_NO_DISCORDANCE")
        self.assertEqual(evidence.deformation_weight, 0.0)

    def test_large_matched_block_can_support_equivalence(self):
        evidence = summarize_decision_evidence(
            "d",
            pairs([(1, 1)] * 20000),
            deformation_normalizer=DeformationNormalizer(scale=1.0, pair_count=20000),
        )
        self.assertTrue(evidence.equivalence_supported)
        self.assertFalse(evidence.unresolved)
        self.assertGreater(evidence.deformation_weight, 0.0)
        self.assertLessEqual(evidence.deformation_weight, 0.1)

    def test_paired_interval_does_not_bootstrap_to_zero_width(self):
        lower, upper = paired_success_difference_interval(0, 0, 32)
        self.assertLess(lower, -0.02)
        self.assertGreater(upper, 0.02)

    def test_deformation_scale_rejects_nontraining_fit(self):
        with self.assertRaisesRegex(ValueError, "training sources only"):
            DeformationNormalizer.fit(pairs([(1, 1)]), split="validation")


class LossTests(unittest.TestCase):
    def evidence(self, r: float, b: float, deformation_weight: float = 0.0) -> DecisionEvidence:
        return DecisionEvidence(
            decision_id="d", n_pairs=32, both_fail=0, rescue=int(r * 32),
            break_count=int(b * 32), both_success=0, r=r, b=b,
            paired_delta=r-b, paired_ci_level=0.9, paired_ci_lower=-0.1,
            paired_ci_upper=0.1, equivalence_margin=0.02,
            equivalence_supported=deformation_weight > 0, status="test", unresolved=False,
            deformation_target=1.0, deformation_weight=deformation_weight,
            mean_jdef0=2.0, mean_jdef1=1.0,
        )

    def test_exact_cost_sensitive_formula(self):
        row = self.evidence(0.25, 0.125)
        score = 0.7
        result = decision_loss_components([score], [row], lambda_break=3.0)
        expected = 0.25 * np.logaddexp(0, -score) + 4.0 * 0.125 * np.logaddexp(0, score)
        self.assertAlmostEqual(result["total"], expected)

    def test_break_penalty_family_and_deformation_separation(self):
        row = self.evidence(0.25, 0.25, deformation_weight=0.05)
        low = decision_loss_components([1.0], [row], lambda_break=0.0)
        high = decision_loss_components([1.0], [row], lambda_break=3.0)
        self.assertGreater(high["success_mean"], low["success_mean"])
        self.assertAlmostEqual(high["deformation_mean"], low["deformation_mean"])
        with self.assertRaisesRegex(ValueError, "selected from"):
            decision_loss_components([0.0], [row], lambda_break=2.0)


class HeadUtilityTests(unittest.TestCase):
    def test_architecture_dimensions_and_normalization(self):
        x = np.arange(4 * 214, dtype=np.float32).reshape(4, 214)
        mean, scale = fit_feature_normalization(x)
        self.assertEqual(mean.shape, (214,))
        self.assertEqual(scale.shape, (214,))
        params = init_head_parameters(17, 214)
        self.assertEqual([tuple(layer["w"].shape) for layer in params], [(214, 64), (64, 64), (64, 1)])
        values = np.asarray(head_logits(params, np.zeros((3, 214), np.float32)))
        self.assertEqual(values.shape, (3,))

    def test_tiny_cpu_training_uses_validation_selection(self):
        # Evidence is deliberately soft rather than converted to hard labels.
        rescue = self.evidence(0.75, 0.0)
        wait = self.evidence(0.0, 0.75)
        rng = np.random.default_rng(5)
        train_x = rng.normal(size=(8, 214)).astype(np.float32)
        val_x = rng.normal(size=(4, 214)).astype(np.float32)
        train_e = [rescue] * 4 + [wait] * 4
        val_e = [rescue] * 2 + [wait] * 2
        params, normalization, history, metadata = train_decision_head(
            train_x, train_e, val_x, val_e, seed=17, lambda_break=1.0, epochs=3
        )
        self.assertEqual(metadata["architecture"], [214, 64, 64, 1])
        self.assertEqual(metadata["normalization_fit_split"], "train")
        self.assertIn(metadata["best_epoch"], (1, 2, 3))
        self.assertEqual(len(history), 3)
        self.assertEqual(normalization["mean"].shape, (214,))
        self.assertEqual(len(params), 3)
        probability = predict_head_probability(params, val_x, normalization)
        raw_logit = predict_head_logit(params, val_x, normalization)
        self.assertEqual(probability.shape, (4,))
        self.assertEqual(raw_logit.shape, (4,))
        self.assertTrue(np.all((probability >= 0.0) & (probability <= 1.0)))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "head.npz")
            save_decision_head_checkpoint(
                path, params, normalization, metadata, threshold_logit=0.25
            )
            saved = np.load(path)
            self.assertEqual(saved["layer_0_weight"].shape, (214, 64))
            self.assertEqual(saved["normalization_mean"].shape, (214,))
            self.assertAlmostEqual(float(saved["threshold_logit"]), 0.25)

    @staticmethod
    def evidence(r: float, b: float) -> DecisionEvidence:
        return DecisionEvidence(
            decision_id="d", n_pairs=32, both_fail=0, rescue=0, break_count=0,
            both_success=0, r=r, b=b, paired_delta=r-b, paired_ci_level=0.9,
            paired_ci_lower=-1, paired_ci_upper=1, equivalence_margin=.02,
            equivalence_supported=False, status="test", unresolved=False,
            deformation_target=.5, deformation_weight=0, mean_jdef0=0, mean_jdef1=0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
