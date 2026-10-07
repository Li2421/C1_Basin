"""Does the old H20 critic use correct fresh-controller response?"""
from __future__ import annotations

import json

import numpy as np

from .fresh_controller import DEST
from .fresh_controller_evaluate import local_h20_context, outcomes, score
from .probe import OUT, load_scene, write


def nll(p, rows):
    s = np.asarray([r["success"] for r in rows], float)
    f = np.asarray([r["failure"] for r in rows], float)
    return float((-s*np.log(np.clip(p,1e-8,1)) - f*np.log(np.clip(1-p,1e-8,1))).sum() / (s+f).sum())


def run():
    data = load_scene("ring_exchange")
    val_ix = np.flatnonzero(data["split"] == "validation")
    rows = outcomes(data)
    correct = local_h20_context(data, val_ix)
    wrong_base = np.asarray(data["d"]["base_c"][data["ids"][val_ix]], np.float32)
    wrong_v9 = np.asarray(data["d"]["alt_c"][data["ids"][val_ix]], np.float32)
    wrong_nominal_only = np.concatenate((wrong_base[:, :10], correct[:, 10:]), axis=1)
    wrong_eta_response_only = np.concatenate((correct[:, :10], wrong_base[:, 10:]), axis=1)
    vector = np.asarray(json.loads((DEST / "v8_fingerprint.json").read_text())["vector"], np.float32)
    results = []
    for seed in (17, 23, 41):
        path = OUT / "models/ring_exchange/physical_context" / f"seed{seed}"
        p = score(path, "physical_context", seed, data, val_ix, vector, correct)
        base = score(path, "physical_context", seed, data, val_ix, vector, wrong_base)
        v9 = score(path, "physical_context", seed, data, val_ix, vector, wrong_v9)
        nominal = score(path, "physical_context", seed, data, val_ix, vector, wrong_nominal_only)
        eta_response = score(path, "physical_context", seed, data, val_ix, vector, wrong_eta_response_only)
        results.append({"seed": seed, "correct_v8_NLL": nll(p, rows),
                        "wrong_base_NLL": nll(base, rows), "wrong_v9_NLL": nll(v9, rows),
                        "wrong_nominal_only_NLL": nll(nominal, rows),
                        "wrong_eta_response_only_NLL": nll(eta_response, rows),
                        "correct_minus_wrong_base_NLL": nll(p,rows)-nll(base,rows),
                        "median_abs_probability_change_wrong_base": float(np.median(np.abs(p-base))),
                        "mean_abs_probability_change_wrong_base": float(np.mean(np.abs(p-base)))})
    write(DEST / "fresh_context_replacement.json", {
        "new_task_rollout": 0, "fresh_v8_labels_used_only_for_posthoc_evaluation": True,
        "controller_context_causally_used_if_correct_context_beats_matched_wrong": True,
        "results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
