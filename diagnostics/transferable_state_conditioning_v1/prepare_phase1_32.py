"""Freeze a 32-state-per-scene confirmation tranche before opening labels.

The states are the first 32 eligible TEST states in the prior label-blind
sampler order. This reduces simulation without outcome-conditioned filtering.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, PANELS, SCENES, read, sha
from diagnostics.transferable_state_conditioning_v1.predict_frozen import target_data, derangement
from diagnostics.transferable_state_conditioning_v1.stage_evaluate_loso import model_logits

ROOT = HERE / "stage_v3_coverage" / "final_phase1_32"
SOURCE_MODELS = HERE / "stage_v3_coverage" / "final_models_frozen.json"
N = 32


def frozen_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    if path.exists():
        assert path.read_text() == content, f"immutable artifact differs: {path}"
    else:
        path.write_text(content)


def main():
    assert os.environ.get("TRANSFER_Q_FINAL_ROOT") == str(ROOT)
    assert os.environ.get("TRANSFER_Q_FINAL_EXPERIMENT") == "exp_transferable_q_phase1_32_20261008_v1"
    parent_split = HERE / "frozen_splits.json"
    split = read(parent_split)
    split["phase1"] = {"parent_frozen_splits_sha256": sha(parent_split),
                       "rule": "first 32 untouched eligible TEST states per scene in frozen sampler order",
                       "per_scene_states": N,
                       "all_scenes_included": True,
                       "expansion_rule": "extend only if prospective tranche cannot resolve the question"}
    for scene in SCENES:
        for key in ("final_confirmation", "anchor_subset"):
            value = split[key][scene]
            value["state_uids"] = value["state_uids"][:N]
            value["count"] = N
            value["rule"] = split["phase1"]["rule"]
        assert split["final_confirmation"][scene]["state_uids"] == split["anchor_subset"][scene]["state_uids"]
    split_path = ROOT / "frozen_splits.json"
    frozen_json(split_path, split)
    models = read(SOURCE_MODELS)
    assert models["final_outcomes_opened"] is False
    models["phase1_parent_source_only_models_sha256"] = sha(SOURCE_MODELS)
    models["frozen_splits_sha256"] = sha(split_path)
    model_path = ROOT / "final_models_frozen.json"
    frozen_json(model_path, models)
    for panel in PANELS:
        scene = panel.removeprefix("within_").removeprefix("loso_")
        info = models["panels"][panel]
        for bank in ("proposal", "anchor"):
            ids, all_x, all_eta = target_data(scene, bank)
            ids = ids[:N]
            x = {k:v[:N] for k,v in all_x.items()}
            eta = all_eta[:N]
            assert ids == split["final_confirmation"][scene]["state_uids"]
            z_state, z_eta, mismatch = model_logits(info, x, eta)
            rng = np.random.default_rng(20261008)
            permutations = np.stack([derangement(N, rng) for _ in range(100)])
            mismatch_choice = np.stack([mismatch(p).argmax(1) for p in permutations]).astype(np.int16)
            out = ROOT / "predictions" / panel
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{bank}.npz"
            if not path.exists():
                np.savez_compressed(path, state_uids=np.asarray(ids), eta=eta,
                                    state_logits=z_state, eta_logits=z_eta,
                                    state_choice=z_state.argmax(1), eta_choice=z_eta.argmax(1),
                                    mismatch_choice=mismatch_choice,
                                    state_permutations=permutations)
            else:
                old = np.load(path)
                assert np.array_equal(old["state_uids"], ids)
                np.testing.assert_allclose(old["state_logits"], z_state, rtol=0, atol=0)
                assert np.array_equal(old["mismatch_choice"], mismatch_choice)
            audit = {"panel": panel, "bank": bank, "states": N,
                     "phase1_rule": split["phase1"]["rule"],
                     "chosen_state_model": info["state_kind"],
                     "final_models_frozen_sha256": sha(model_path),
                     "predictions_sha256": sha(path),
                     "final_outcomes_opened": False,
                     "selection_change_rate_under_100_derangements": float(np.mean(
                         mismatch_choice != z_state.argmax(1)[None, :]))}
            frozen_json(out / f"{bank}_audit.json", audit)
            print(panel, bank, audit["selection_change_rate_under_100_derangements"], flush=True)
    from diagnostics.transferable_state_conditioning_v1 import prepare_final
    assert prepare_final.HERE == ROOT
    prepare_final.prepare()


if __name__ == "__main__":
    main()
