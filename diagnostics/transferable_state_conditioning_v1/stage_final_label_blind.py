"""Inspect frozen selectors on untouched TEST states without opening outcomes.

For any rollout success in [0,1], the matched-minus-permuted decision value
cannot exceed the fraction of states whose selected candidate changes.
"""
from __future__ import annotations

import json

import numpy as np

from diagnostics.transferable_state_conditioning_v1.assemble_source import PANELS, read, sha
from diagnostics.transferable_state_conditioning_v1.predict_frozen import target_data, derangement
from diagnostics.transferable_state_conditioning_v1.stage_evaluate_loso import model_logits
from diagnostics.transferable_state_conditioning_v1.stage_source import STAGE


def main():
    model_file = STAGE / "final_models_frozen.json"
    models = read(model_file)
    assert models["final_outcomes_opened"] is False
    summary = {"schema": "frozen_final_label_blind_switch_capacity_v1",
               "source_models_sha256": sha(model_file),
               "final_outcome_labels_read": False, "panels": {}}
    for panel in PANELS:
        scene = panel.removeprefix("within_").removeprefix("loso_")
        info = models["panels"][panel]
        summary["panels"][panel] = {}
        for bank in ("proposal", "anchor"):
            ids, x, eta = target_data(scene, bank)
            z_state, z_eta, mismatch = model_logits(info, x, eta)
            rng = np.random.default_rng(20261008)
            perm = np.stack([derangement(len(ids), rng) for _ in range(100)])
            choices = np.stack([mismatch(p).argmax(1) for p in perm])
            selected = z_state.argmax(1)
            rate = float(np.mean(choices != selected[None, :]))
            out = STAGE / "final_label_blind" / panel
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{bank}.npz"
            np.savez_compressed(path, state_uids=np.asarray(ids), eta=eta,
                                state_logits=z_state, eta_logits=z_eta,
                                state_choice=selected, eta_choice=z_eta.argmax(1),
                                state_permutations=perm, mismatch_choice=choices)
            summary["panels"][panel][bank] = {
                "independent_states": len(ids),
                "same_scene_permutations": len(perm),
                "selection_change_rate": rate,
                "maximum_possible_matched_minus_mismatch_mean_success": rate,
                "state_choice_counts": np.bincount(selected, minlength=16).tolist(),
                "eta_choice_counts": np.bincount(z_eta.argmax(1), minlength=16).tolist(),
                "predictions_sha256": sha(path)}
            print(panel, bank, rate, flush=True)
    path = STAGE / "final_label_blind" / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
