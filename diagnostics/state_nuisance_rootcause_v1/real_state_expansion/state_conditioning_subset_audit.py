"""Read-only, seed-split evaluation of states needing eta conditioning.

The TRAIN-selected eta-only policy is frozen before TEST is read. A physical
state is selected using eight TEST rollout seeds only when another eta has a
success-rate lower bound at least 0.25 above the fixed eta's upper bound.
All policy scores for that selected set use the other eight seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEEDS = (17, 23, 41)
ARMS = ("physical23_eta_n32", "physical23_eta_n512", "physical23_heads_n512")
FOLDS = ((0, 8, 8, 16), (8, 16, 0, 8))


def interval(x: np.ndarray, *, bootstrap_seed: int) -> list[float]:
    """Paired physical-state bootstrap of a mean; fit seeds are averaged first."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.integers(0, len(x), size=(20000, len(x)))
    return np.quantile(x[sampled].mean(axis=1), [0.025, 0.975]).tolist()


def half_counts(outcomes: np.ndarray, lo: int, hi: int):
    sl = outcomes[:, :, lo:hi]
    success = sl[:, :, :, 0].sum(axis=-1).astype(int)
    unknown = sl[:, :, :, 1].sum(axis=-1).astype(int)
    n = hi - lo
    failure = n - success - unknown
    assert np.all(failure >= 0)
    return success, failure, unknown


def main() -> None:
    data = np.load(ROOT / "fresh_test_dataset.npz")
    logits = np.load(ROOT / "evaluation_logits.npz")
    frozen = json.loads((ROOT / "models_frozen.json").read_text())["selections"]
    outcomes = data["rollout_outcomes"]
    assert outcomes.shape == (48, 16, 16, 3)

    eta_prior = logits["eta_only_n512__fresh_test"]
    fixed = int(np.argmax(eta_prior[0]))
    assert fixed == 13 and np.all(np.argmax(eta_prior, axis=1) == fixed)
    picks = {}
    for arm in ARMS:
        step = frozen[arm]["selected_step"]
        predictions = np.stack(
            [logits[f"{arm}__{seed}__{step}__fresh_test"] for seed in SEEDS]
        )
        assert predictions.shape == (3, 48, 16)
        picks[arm] = predictions.argmax(axis=-1)

    full_success, full_failure, full_unknown = half_counts(outcomes, 0, 16)
    fixed_full = full_success[:, fixed] / 16
    report = {
        "definition": "Discovery half: some eta success lower bound minus TRAIN-selected fixed eta success upper bound >= 0.25; evaluation uses the independent other eight seeds.",
        "fixed_eta_index": fixed,
        "model_seeds": list(SEEDS),
        "bootstrap": "20,000 paired resamples of distinct physical states; average three fit seeds within each resample",
        "full_fresh_test": {
            "states": 48,
            "fixed_eta_success_lower": float(fixed_full.mean()),
            "fixed_eta_success_upper": float(((16 - full_failure[:, fixed]) / 16).mean()),
            "best_fixed_eta_by_full_test_lower_descriptive": int(
                np.argmax(full_success.mean(axis=0))
            ),
            "models": {},
        },
        "folds": [],
    }
    for arm, choice in picks.items():
        picked = full_success[np.arange(48)[None, :], choice] / 16
        delta = picked - fixed_full[None, :]
        report["full_fresh_test"]["models"][arm] = {
            "selected_eta_success_lower_by_fit": picked.mean(axis=1).tolist(),
            "delta_vs_fixed_by_fit": delta.mean(axis=1).tolist(),
            "delta_vs_fixed_mean": float(delta.mean()),
            "delta_vs_fixed_state_bootstrap_95": interval(delta.mean(axis=0), bootstrap_seed=747),
            "eta_choices_by_fit": [np.bincount(c, minlength=16).tolist() for c in choice],
        }

    for fold, (disc_lo, disc_hi, eval_lo, eval_hi) in enumerate(FOLDS):
        ds, df, du = half_counts(outcomes, disc_lo, disc_hi)
        n_disc = disc_hi - disc_lo
        discovery_gain_lower = ds / n_disc - (n_disc - df[:, fixed, None]) / n_disc
        # The threshold is robust even if every unknown on the alternative is a
        # failure and every unknown on the fixed eta is a success.
        eligible = np.flatnonzero(discovery_gain_lower.max(axis=1) >= 0.25)
        es, ef, eu = half_counts(outcomes, eval_lo, eval_hi)
        n_eval = eval_hi - eval_lo
        fixed_lower = es[eligible, fixed] / n_eval
        fixed_upper = (n_eval - ef[eligible, fixed]) / n_eval
        fold_result = {
            "discovery_seed_range": [disc_lo, disc_hi],
            "evaluation_seed_range": [eval_lo, eval_hi],
            "state_count": int(len(eligible)),
            "state_indices": data["state_indices"][eligible].astype(int).tolist(),
            "fixed_eta_eval_success_lower": float(fixed_lower.mean()),
            "fixed_eta_eval_success_upper": float(fixed_upper.mean()),
            "models": {},
        }
        for arm, choice in picks.items():
            choice = choice[:, eligible]
            picked_success = es[eligible[None, :], choice]
            picked_failure = ef[eligible[None, :], choice]
            picked_lower = picked_success / n_eval
            picked_upper = (n_eval - picked_failure) / n_eval
            gain_point = picked_lower - fixed_lower[None, :]
            gain_lower = picked_lower - fixed_upper[None, :]
            gain_upper = picked_upper - fixed_lower[None, :]
            same = choice == fixed
            gain_lower[same] = gain_upper[same] = 0
            assert np.all(gain_lower <= gain_point + 1e-12)
            assert np.all(gain_point <= gain_upper + 1e-12)
            fold_result["models"][arm] = {
                "selected_eta_by_fit": choice.tolist(),
                "eval_success_lower_by_fit": picked_lower.mean(axis=1).tolist(),
                "gain_vs_fixed_by_fit": gain_point.mean(axis=1).tolist(),
                "gain_vs_fixed_mean": float(gain_point.mean()),
                "gain_vs_fixed_mean_unknown_bounds": [
                    float(gain_lower.mean()), float(gain_upper.mean())
                ],
                "gain_vs_fixed_state_bootstrap_95": interval(
                    gain_point.mean(axis=0), bootstrap_seed=748 + fold
                ),
                "positive_equal_negative_by_fit": [
                    [int((g > 0).sum()), int((g == 0).sum()), int((g < 0).sum())]
                    for g in gain_point
                ],
            }
        report["folds"].append(fold_result)

    destination = ROOT / "state_conditioning_subset_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
