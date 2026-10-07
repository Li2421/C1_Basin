from __future__ import annotations

import numpy as np

from shared_control.basis_families import get_basis_family
from shared_control.diagnostic_corrector import DiagnosticCorrector
from diagnostics.double_bottleneck_eta_basis_redesign.tools.bases import (
    basis_terms as archived_basis_terms,
    correction as archived_correction,
)


def test_p0_matches_legacy_and_archived_implementation() -> None:
    rng = np.random.default_rng(9182)
    family = get_basis_family("p0")
    for n_agents in (2, 4):
        for _ in range(20):
            positions = rng.normal(size=(n_agents, 2))
            goals = rng.normal(size=(n_agents, 2))
            u_safe = rng.normal(size=(n_agents, 2))
            eta = rng.normal(size=3)
            actual = family.compute(positions, goals, u_safe, 0.5).correction(eta)
            legacy = DiagnosticCorrector(eta)(positions, goals, u_safe, 0.5)[0]
            archived = archived_correction(
                "P0-3D", eta, positions, goals, u_safe, 0.5, 3.303687238760696
            )
            assert np.array_equal(actual, legacy)
            assert np.array_equal(actual, archived)


def test_orthoflow3_matches_archived_implementation() -> None:
    rng = np.random.default_rng(9183)
    family = get_basis_family("orthoflow3")
    for n_agents in (2, 4):
        for _ in range(20):
            positions = rng.normal(size=(n_agents, 2))
            goals = rng.normal(size=(n_agents, 2))
            u_safe = rng.normal(size=(n_agents, 2))
            eta = rng.normal(size=3)
            fields = family.compute(positions, goals, u_safe, 0.5)
            expected_fields = archived_basis_terms(
                "P1-OrthoFlow3", positions, goals, u_safe, 0.5, 3.303687238760696
            )
            for actual, expected in zip(fields.values, expected_fields, strict=True):
                assert np.array_equal(actual, expected)
            assert np.array_equal(
                fields.correction(eta),
                archived_correction(
                    "P1-OrthoFlow3", eta, positions, goals, u_safe, 0.5, 3.303687238760696
                ),
            )


def test_zero_eta_is_exact_zero_for_both_families() -> None:
    rng = np.random.default_rng(9184)
    for name in ("p0", "orthoflow3"):
        fields = get_basis_family(name).compute(
            rng.normal(size=(4, 2)),
            rng.normal(size=(4, 2)),
            rng.normal(size=(4, 2)),
            0.5,
        )
        assert np.array_equal(fields.correction(np.zeros(3)), np.zeros((4, 2)))
