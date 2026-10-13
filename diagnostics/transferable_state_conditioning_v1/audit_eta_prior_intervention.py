"""Posthoc control: repair target ETA marginals, preserve frozen state effects.

This intentionally uses target discovery labels for an oracle diagnostic.
It is NOT a source-only LOSO model or a proposed deployable repair. Per-eta
logit offsets are fitted only to aggregate discovery-half success counts;
evaluation seeds are disjoint. No within-state target label fits an offset.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE / "stage_v3_coverage/final_phase1_32"
OUT = HERE / "posthoc_cause_audit"
SCENES = ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange")


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z, -60, 60)))


def bootstrap(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(20261009)
    b = x[rng.integers(len(x), size=(20000, len(x)))].mean(1)
    return {"mean_difference": float(x.mean()), "ci95": np.quantile(b, [.025, .975]).tolist(),
            "physical_states": len(x)}


def target_marginal_offsets(z, successes):
    # Fixed Jeffreys smoothing keeps all-zero/all-one discovery marginals finite.
    target = (successes.sum((0, 2))+.5)/(successes.shape[0]*successes.shape[2]+1.)
    lo, hi = np.full(16, -100.), np.full(16, 100.)
    for _ in range(80):
        mid = (lo+hi)/2
        below = sigmoid(z+mid).mean(0) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    offsets = (lo+hi)/2
    assert np.allclose(sigmoid(z+offsets).mean(0), target, atol=1e-10)
    return offsets


def metrics(z, q, best_disc):
    n = len(q)
    choice = z.argmax(1)
    selected = q[np.arange(n), choice]
    mismatched = (q[:, choice].sum(1)-selected)/(n-1)
    fixed = q[:, best_disc]
    return {"selection_counts": np.bincount(choice, minlength=16).tolist(),
            "selected_mean_eval_success": float(selected.mean()),
            "mismatch_mean_eval_success": float(mismatched.mean()),
            "matched_minus_mismatch": bootstrap(selected-mismatched),
            "matched_minus_discovery_best_fixed": bootstrap(selected-fixed)}


def main():
    OUT.mkdir(exist_ok=True)
    result = {"schema": "posthoc_target_marginal_oracle_intervention_v1", "new_rollouts": 0,
              "target_labels_used": "discovery seeds 0..7, aggregate per eta only",
              "evaluation_labels": "seeds 8..15",
              "source_only_loso_claim_allowed": False,
              "state_interaction_change": "none; one constant logit offset per eta",
              "panels": {}}
    for scene in SCENES:
        out = np.load(FINAL / "final/anchor" / scene / "outcomes.npz")
        values = out["values"]
        disc = values[:, :, :8, 0]
        q = values[:, :, 8:, 0].mean(2)
        best_disc = int(disc.mean((0, 2)).argmax())
        for prefix in ("within_", "loso_"):
            panel = prefix+scene
            pred = np.load(FINAL / "predictions" / panel / "anchor.npz")
            assert pred["state_uids"].tolist() == out["state_uids"].tolist()
            z = pred["state_logits"].astype(float)
            offsets = target_marginal_offsets(z, disc)
            fixed = z+offsets
            entry = {"best_fixed_discovery_index": best_disc,
                     "best_fixed_eval_success": float(q[:, best_disc].mean()),
                     "original": metrics(z, q, best_disc),
                     "target_marginal_oracle": metrics(fixed, q, best_disc),
                     "per_eta_logit_offsets": offsets.tolist(),
                     "numerical_seed_count_discovery": int(values[:, :, :8, 4].sum()),
                     "numerical_seed_count_evaluation": int(values[:, :, 8:, 4].sum())}
            result["panels"][panel] = entry
            np.savez_compressed(OUT / (panel+"_prior_intervention.npz"),
                                state_uids=out["state_uids"], eta=out["eta"],
                                original_logits=z, marginal_repaired_logits=fixed,
                                logit_offsets=offsets, evaluation_success=q)
            print(panel, "original", entry["original"]["selected_mean_eval_success"],
                  "marginal_repaired", entry["target_marginal_oracle"]["selected_mean_eval_success"],
                  "fixed", entry["best_fixed_eval_success"],
                  "repaired_mismatch_gap", entry["target_marginal_oracle"]["matched_minus_mismatch"], flush=True)
    (OUT / "prior_intervention.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
