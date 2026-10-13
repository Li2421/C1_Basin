"""Read-only Part 3 analysis of frozen Ring P1 Flow controller labels/features.

No simulator calls, training, or outcome-dependent proposal generation.
Run with the repository's NumPy environment. Unknown outcomes remain intervals.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P1 = ROOT / "diagnostics/controller_q_transfer_round3_v1"


def mean_interval(y, u, axis=-1):
    """Unknown is neither a success nor a failure."""
    return np.mean(y, axis=axis), np.mean(y + u, axis=axis)


def selection_value(q):
    """Statewise value for a uniform pair prior, shared K16 candidate bank."""
    return 0.5 * (q[0].max(axis=-1) + q[1].max(axis=-1)) - (0.5 * (q[0] + q[1])).max(axis=-1)


def boot_interval(values, rng, repetitions=5000):
    values = np.asarray(values, float)
    draws = rng.integers(0, len(values), (repetitions, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()


def main():
    path = P1 / "P1_LABELS.npz"
    with np.load(path) as z:
        y, u = z["Y"].copy(), z["U"].copy()
        names, states = z["controller_names"].tolist(), z["state_uids"].tolist()
    assert y.shape == u.shape == (14, 24, 16, 4)
    assert np.all((y == 0) | (y == 1)) and np.all((u == 0) | (u == 1))
    assert np.all(y + u <= 1)
    qlo, qhi = mean_interval(y, u)
    # Discovery and confirmation are seed-disjoint. The 24 states are opened
    # development states, so confirmation is NOT a new-state confirmation.
    discover = y[..., :2].mean(axis=-1)
    confirm_lo, confirm_hi = mean_interval(y[..., 2:], u[..., 2:])
    rng = np.random.default_rng(20261012)
    pairs = []
    for c in range(len(names)):
        for d in range(c + 1, len(names)):
            q = discover[[c, d]]
            v_discover = selection_value(q).mean()
            qc = confirm_lo[[c, d]]
            v = selection_value(qc)
            qcu = confirm_hi[[c, d]]
            # Pointwise lower bound under arbitrary unknown assignments.
            v_lower = np.maximum(0, 0.5 * (qc[0].max(-1) + qc[1].max(-1))
                                 - (0.5 * (qcu[0] + qcu[1])).max(-1))
            qdiff_lower = np.maximum(0, np.maximum(qc[0]-qcu[1], qc[1]-qcu[0]))
            uncertainty = np.max(confirm_hi[[c, d]] - qc, axis=(0, 2))
            q_diff = np.abs(qc[0] - qc[1])
            best_discover = np.argmax(q, axis=-1)  # controller x state
            chosen = np.take_along_axis(qc, best_discover[..., None], axis=-1)[..., 0]
            # Cross-use regret uses the OTHER controller's discovery choice.
            cross = np.stack((
                np.take_along_axis(qc[0], best_discover[1, :, None], axis=-1)[:, 0],
                np.take_along_axis(qc[1], best_discover[0, :, None], axis=-1)[:, 0],
            ))
            own_minus_cross = (chosen - cross).mean(axis=0)
            cross_upper = np.stack((
                np.take_along_axis(qcu[0], best_discover[1, :, None], axis=-1)[:, 0],
                np.take_along_axis(qcu[1], best_discover[0, :, None], axis=-1)[:, 0],
            ))
            own_minus_cross_lower = (chosen - cross_upper).mean(axis=0)
            f1 = np.load(P1 / "p1_features" / (names[c] + ".npz"))
            f2 = np.load(P1 / "p1_features" / (names[d] + ".npz"))
            assert f1["state_uids"].tolist() == states == f2["state_uids"].tolist()
            r3 = np.linalg.norm(f1["R3"] - f2["R3"], axis=1)
            r4 = np.linalg.norm(f1["R4_action"] - f2["R4_action"], axis=1)
            f1.close(); f2.close()
            pairs.append(dict(c=names[c], d=names[d], discovery_V=float(v_discover),
                confirmation_V=float(v.mean()), confirmation_V_state_bootstrap_95=boot_interval(v, rng),
                confirmation_V_unknown_safe_lower=float(v_lower.mean()),
                confirmation_unknown_max_width=float(uncertainty.mean()),
                confirmation_Q_abs_mean=float(q_diff.mean()),
                confirmation_Q_abs_unknown_safe_lower=float(qdiff_lower.mean()),
                confirmation_Q_abs_max=float(q_diff.max()),
                discovery_choice_different_states=int(np.sum(best_discover[0] != best_discover[1])),
                confirmation_own_minus_cross=float(own_minus_cross.mean()),
                confirmation_own_minus_cross_unknown_safe_lower=float(own_minus_cross_lower.mean()),
                confirmation_own_minus_cross_bootstrap_95=boot_interval(own_minus_cross, rng),
                R3_mean_distance=float(r3.mean()), R4_action_mean_distance=float(r4.mean()),
                R3_min_distance=float(r3.min()), R4_action_min_distance=float(r4.min())))
    pairs.sort(key=lambda x: x["discovery_V"], reverse=True)
    selected = pairs[0]
    # Pair selection on discovery seeds; confirmation estimates should not be
    # described as independent physical-state validation.
    def corr(a, b):
        a=np.asarray(a); b=np.asarray(b)
        return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
    summary = dict(
        source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        setting="Ring FF, same 24 opened true-t0 states, 14 real Flow checkpoints, common K16, Q4; seed0-1 discovery / seed2-3 confirmation",
        pairs=91, unknown_cells=int(u.sum()),
        any_confirmed_Q_difference_pairs=int(sum(p["confirmation_Q_abs_mean"] > 0 for p in pairs)),
        discovery_top_pair=selected,
        positive_confirmation_V_pairs=int(sum(p["confirmation_V"] > 0 for p in pairs)),
        positive_confirmation_cross_use_pairs=int(sum(p["confirmation_own_minus_cross"] > 0 for p in pairs)),
        R3_vs_Q_abs_corr=corr([p["R3_mean_distance"] for p in pairs],[p["confirmation_Q_abs_mean"] for p in pairs]),
        R4_vs_Q_abs_corr=corr([p["R4_action_mean_distance"] for p in pairs],[p["confirmation_Q_abs_mean"] for p in pairs]),
        R3_vs_V_corr=corr([p["R3_mean_distance"] for p in pairs],[p["confirmation_V"] for p in pairs]),
        R4_vs_V_corr=corr([p["R4_action_mean_distance"] for p in pairs],[p["confirmation_V"] for p in pairs]),
        note="R3/R4 distances are sampled descriptors, not certified behavior suprema. Q2 confirmation is noisy and states are opened development states.",
        new_rollouts=0, new_training=0)
    (HERE / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (HERE / "pairs.json").write_text(json.dumps(pairs, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
