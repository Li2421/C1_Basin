"""Cheap DEVELOPMENT LOSO audit on exact cached Q4 source/probe labels.

Every fold uses a source-only frozen pilot critic. Target scene labels here
are exploratory and never eligible for final confirmation.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from diagnostics.physical_state_critic_formal_v1 import physical_entities
from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, FORMAL, SCENES, read, sha
from diagnostics.transferable_state_conditioning_v1.evaluate_final import paired_bootstrap, safe_corr
from diagnostics.transferable_state_conditioning_v1.predict_frozen import logits, derangement
from diagnostics.transferable_state_conditioning_v1.predict_selected import residual_logits
from diagnostics.transferable_state_conditioning_v1.stage_source import STAGE, scene_rows


def model_logits(info, x, eta):
    eta_info = {"seed": info["eta_seed"], "checkpoint": info["eta_checkpoint"],
                "checkpoint_sha256": info["eta_checkpoint_sha256"]}
    z_eta = logits(eta_info, "eta_only_wide", x, eta)
    if info["state_kind"] == "state_concat":
        state_info = {"seed": info["state_seed"], "checkpoint": info["state_checkpoint"],
                      "checkpoint_sha256": info["state_checkpoint_sha256"]}
        z_state = logits(state_info, "state_concat", x, eta)
        mismatched = lambda mapping: logits(state_info, "state_concat", x, eta, mapping)
    else:
        assert info["state_kind"] == "state_residual"
        features, apply = residual_logits(info, x, eta)
        z_state = z_eta + np.asarray(apply(features))
        mismatched = lambda mapping: z_eta + np.asarray(apply(
            {k: v[mapping] for k, v in features.items()}))
    return z_state, z_eta, mismatched


def audit(scene, bank, frozen, variant="selected"):
    panel = f"loso_{scene}"
    info = frozen["panels"][panel]
    if bank == "proposal":
        target_indices = read(HERE / "stage_v1" / "source_index_splits.json")["selection"][scene]
        rows = scene_rows(scene, target_indices)
        physical = [r["physical"] for r in rows]
        state_uids = [r["state_uid"] for r in rows]
        eta = np.stack([r["eta"] for r in rows])
        q = np.stack([r["success"] for r in rows]).astype(float) / 4.
    else:
        assert bank == "anchor"
        source = {r["alias"]: r for r in read(FORMAL / f"states_{scene}.json")["states"]}
        proposals = read(FORMAL / "loso" / scene / "train" / "proposals.json")["states"][:16]
        physical = [source[p["state_alias"]]["physical"] for p in proposals]
        state_uids = [p["state_uid"] for p in proposals]
        eta = np.broadcast_to(np.asarray(read(HERE / "frozen_splits.json")["common_anchor_eta"],
                                         np.float32), (16, 16, 3)).copy()
        signal = read(HERE / "source_anchor_probe" / "signal.json")["scenes"][scene]
        q = np.asarray(signal["success_counts"], float) / 4.
    assert eta.shape == (len(q), 16, 3)
    assert len(set(state_uids)) == len(q)
    x = physical_entities.batch([{"scene": scene, "physical": p} for p in physical])
    z_state, z_eta, mismatch = model_logits(info, x, eta)
    n = len(q); row = np.arange(n)
    state_choice, eta_choice = z_state.argmax(1), z_eta.argmax(1)
    rng = np.random.default_rng(20261008)
    permutations = np.stack([derangement(n, rng) for _ in range(100)])
    mismatch_choice = np.stack([mismatch(p).argmax(1) for p in permutations])
    selected = q[row, state_choice]
    eta_selected = q[row, eta_choice]
    mismatch_q = q[row[None,:], mismatch_choice]
    result = {"schema": "source_only_development_loso_Q4_v1", "panel": panel,
              "bank": bank, "independent_target_states": n,
              "source_only_models_sha256": sha(STAGE / "final_models_frozen.json"),
              "target_labels_used_for_model_fit_or_checkpoint": False,
              "target_labels_are_final_confirmation": False,
              "Q4_all_success_count": {
                  "Q_state": int(np.sum(selected == 1)),
                  "Q_eta": int(np.sum(eta_selected == 1)),
                  "mismatched_state_mean": float(np.mean(np.sum(mismatch_q == 1, axis=1))),
                  "Oracle": int(np.sum(q.max(1) == 1))},
              "mean_success": {"Q_state": float(selected.mean()),
                               "Q_eta": float(eta_selected.mean()),
                               "mismatched_state": float(mismatch_q.mean()),
                               "Oracle": float(q.max(1).mean())},
              "paired_state_bootstrap": {
                  "matched_minus_eta": paired_bootstrap(selected - eta_selected),
                  "matched_minus_mismatch": paired_bootstrap(selected - mismatch_q.mean(0)),
                  "matched_minus_eta_4of4": paired_bootstrap(
                      (selected == 1).astype(float) - (eta_selected == 1).astype(float)),
                  "matched_minus_mismatch_4of4": paired_bootstrap(
                      (selected == 1).astype(float) - (mismatch_q == 1).mean(0))},
              "selection_changed_by_state_permutation": float(np.mean(
                  mismatch_choice != state_choice[None,:])),
              "state_choice_index_counts": np.bincount(state_choice, minlength=16).tolist(),
              "eta_choice_index_counts": np.bincount(eta_choice, minlength=16).tolist()}
    if bank == "anchor":
        fixed_best = int(q.mean(0).argmax())
        posthoc = q.mean(0)
        boot_rng = np.random.default_rng(20261008)
        samples = boot_rng.integers(n, size=(20000,n))
        diff = selected[samples].mean(1) - q[samples].mean(1).max(1)
        true_effect = q - q.mean(0, keepdims=True)
        pred = 1/(1+np.exp(-np.clip(z_state,-50,50)))
        pred_effect = pred - pred.mean(0, keepdims=True)
        a,b = np.triu_indices(16,1)
        truth = q[:,a]-q[:,b]
        reversal = np.array([np.any(truth[:,i]>=.5) and np.any(truth[:,i]<=-.5)
                             for i in range(len(a))])
        cases = (np.abs(truth)>=.5) & reversal[None,:]
        correct_state = np.sign(z_state[:,a]-z_state[:,b]) == np.sign(truth)
        correct_eta = np.sign(z_eta[:,a]-z_eta[:,b]) == np.sign(truth)
        result["fixed_eta"] = {"all_16_mean_success": posthoc.tolist(),
                               "posthoc_best_index": fixed_best,
                               "posthoc_best_mean_success": float(posthoc.max()),
                               "state_minus_posthoc_best": paired_bootstrap(selected-q[:,fixed_best]),
                               "state_minus_bootstrap_reselected_best": {
                                   "difference": float(selected.mean()-posthoc.max()),
                                   "ci95": np.quantile(diff,[.025,.975]).tolist()},
                               "oracle_minus_best_fixed": float(q.max(1).mean()-posthoc.max())}
        result["mechanism"] = {"state_effect_correlation": safe_corr(pred_effect,true_effect),
                                "strong_reversal_eta_pairs": int(reversal.sum()),
                                "strong_pair_state_cases": int(cases.sum()),
                                "switch_accuracy_state": float(correct_state[cases].mean()) if cases.any() else None,
                                "switch_accuracy_eta": float(correct_eta[cases].mean()) if cases.any() else None,
                                "diagnostic_seed_limit": "same four Q4 seeds; final audit requires independent 8+8 halves"}
    outdir = (STAGE / "development_loso" / scene if variant == "selected"
              else STAGE / "development_loso_ablations" / variant / scene)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"{bank}_predictions.npz",
                        state_uids=np.asarray(state_uids), eta=eta,
                        q4=q, state_logits=z_state, eta_logits=z_eta,
                        state_choice=state_choice, eta_choice=eta_choice,
                        mismatch_choice=mismatch_choice,
                        state_permutations=permutations)
    (outdir / f"{bank}_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    frozen = read(STAGE / "final_models_frozen.json")
    for scene in SCENES:
        for bank in ("proposal", "anchor"):
            result = audit(scene, bank, frozen)
            print(scene, bank, result["mean_success"],
                  result["paired_state_bootstrap"]["matched_minus_mismatch"], flush=True)
