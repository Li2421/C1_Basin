import numpy as np

from new_benchmark_common.safety_eta3 import (
    DOMAIN_HIGH,
    DOMAIN_LOW,
    _robust_decision_from_rows,
    designs,
    promote,
)


def test_frozen_common_sobol_design_is_deterministic_and_in_domain():
    first = designs()
    second = designs()
    assert first == second
    for name in ("primary", "secondary"):
        points = np.asarray(first[name])
        assert points.shape == (256, 3)
        assert np.all(points >= DOMAIN_LOW)
        assert np.all(points <= DOMAIN_HIGH)
    assert not np.array_equal(np.asarray(first["primary"]), np.asarray(first["secondary"]))


def test_promotion_keeps_every_perfect_screen_point():
    points = np.asarray(designs()["primary"])
    q4 = {index: (4 if index < 20 else 3 if index < 30 else 0) for index in range(256)}
    selected = promote(points, q4)
    assert selected == list(range(20))


def test_promotion_caps_three_of_four_candidates_at_sixteen_total():
    points = np.asarray(designs()["primary"])
    q4 = {index: (4 if index < 3 else 3 if index < 40 else 0) for index in range(256)}
    selected = promote(points, q4)
    assert selected[:3] == [0, 1, 2]
    assert len(selected) == 16
    assert all(q4[index] >= 3 for index in selected)


def test_exact_robust_rejection_stops_at_second_observed_failure():
    rows = {0: {"success": 1}, 1: {"success": 0},
            2: {"success": 1}, 3: {"success": 0}}
    decision = _robust_decision_from_rows(rows)
    assert decision["stop_reason"] == "ROBUST_IMPOSSIBLE_2_FAILURES"
    assert decision["evaluated_seed_count"] == 4
    assert decision["n_success"] == 2
    assert decision["n_failure"] == 2
    assert decision["unrun_seeds"] == list(range(4, 16))
    assert not decision["robust"]


def test_exact_robust_acceptance_requires_fifteen_successes():
    fourteen = {seed: {"success": 1} for seed in range(14)}
    assert _robust_decision_from_rows(fourteen) is None
    fifteen = {seed: {"success": 1} for seed in range(15)}
    decision = _robust_decision_from_rows(fifteen)
    assert decision["stop_reason"] == "ROBUST_CONFIRMED_15_SUCCESSES"
    assert decision["robust"]
    assert decision["unrun_seeds"] == [15]
    assert _robust_decision_from_rows(
        fifteen, allow_early_acceptance=False) is None
