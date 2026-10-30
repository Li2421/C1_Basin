import numpy as np

from new_benchmark_common.basin_dataset_v1 import (
    _canonical_center,
    _robust_components,
    content_hash,
)


def _label(eta, uid, *, robust=True, seeds=16, successes=15):
    return {
        "eta_raw": list(eta),
        "eta_normalized": list(eta),
        "eta_uid": uid,
        "robust_15of16": robust,
        "seed_count": seeds,
        "success_count": successes,
    }


def test_content_hash_is_key_order_invariant():
    assert content_hash({"a": 1, "b": [2, 3]}) == content_hash(
        {"b": [2, 3], "a": 1})


def test_canonical_center_prefers_stronger_evidence_then_small_norm():
    labels = [
        _label([0.2, 0.0, 0.0], "a", seeds=16, successes=15),
        _label([0.1, 0.0, 0.0], "b", seeds=32, successes=30),
        _label([0.0, 0.0, 0.0], "c", robust=False, seeds=16, successes=14),
    ]
    assert _canonical_center(labels)["eta_uid"] == "b"


def test_components_never_include_nonrobust_points_and_cap_at_three():
    labels = [
        _label([-0.45, -0.45, -0.45], "a"),
        _label([-0.1, -0.1, -0.1], "b"),
        _label([0.1, 0.1, 0.1], "c"),
        _label([0.45, 0.45, 0.45], "d"),
        _label([0.0, 0.0, 0.0], "negative", robust=False, successes=14),
    ]
    components = _robust_components(labels)
    assert len(components) <= 3
    assert "negative" not in {row["eta_uid"] for group in components for row in group}
    assert all(np.isfinite(row["eta_normalized"]).all()
               for group in components for row in group)
