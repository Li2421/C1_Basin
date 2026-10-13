"""Cache-only joint four-scene fits; source validation and fresh-state freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "joint_four_scene_v1"
STAGE = HERE / "stage_v3_coverage"
PANEL = "joint_four_scenes"
SCENES = ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if path.exists():
        assert path.read_text() == content, path
    else:
        path.write_text(content)


def prepare():
    arrays = []
    sources = {}
    for scene in SCENES:
        path = STAGE / "source_data" / ("within_"+scene) / "trainval.npz"
        manifest = read(path.parent / "manifest.json")
        assert sha(path) == manifest["trainval_sha256"]
        arrays.append(dict(np.load(path)))
        sources[scene] = {"path": str(path), "sha256": sha(path)}
    keys = arrays[0].keys()
    assert all(a.keys() == keys for a in arrays)
    merged = {}
    for k in keys:
        xs = [a[k] for a in arrays]
        if k.startswith("entity_"):
            shape = tuple(max(x.shape[j] for x in xs) for j in range(1, xs[0].ndim))
            xs = [np.pad(x, [(0, 0)]+[(0, want-have) for have, want in zip(x.shape[1:], shape)]) for x in xs]
        merged[k] = np.concatenate(xs)
    assert len(set(merged["state_uids"])) == len(merged["state_uids"])
    assert np.all(merged["success"]+merged["failure"]+merged["numerical"] == 4)
    assert np.all(merged["entity_agents"][..., 3:7] == 0)
    assert np.all(merged["entity_agents"][..., 10:15] == 0)
    assert np.all(merged["entity_globals"] == 0)
    out = ROOT / "source_data" / PANEL
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trainval.npz"
    if path.exists():
        old = np.load(path)
        assert all(np.array_equal(old[k], v) for k, v in merged.items())
    else:
        np.savez_compressed(path, **merged)
    counts = {s: {sp: int(np.sum((merged["scene"] == s) & (merged["split"] == sp)))
                  for sp in ("train", "validation")} for s in SCENES}
    freeze(out / "manifest.json", {
        "schema": "joint_four_scene_existing_labels_v1", "panel": PANEL,
        "source_scenes": list(SCENES), "source_counts": counts, "source_artifacts": sources,
        "train_states": int(np.sum(merged["split"] == "train")),
        "validation_states": int(np.sum(merged["split"] == "validation")),
        "trainval_sha256": sha(path), "rollout_seeds_per_state_eta": 4,
        "feature_code_sha256": sha(HERE.parent / "physical_state_critic_formal_v1/physical_entities.py"),
        "test_labels_opened": False, "new_training_rollouts": 0,
        "sample_weighting": "equal contribution from each scene, identical for both critics"})
    original = read(HERE / "frozen_splits.json")
    opened = read(STAGE / "final_phase1_32/frozen_splits.json")
    confirmation = {}
    for scene in SCENES:
        seen = set(opened["final_confirmation"][scene]["state_uids"])
        ids = [u for u in original["final_confirmation"][scene]["state_uids"] if u not in seen][:32]
        assert len(ids) == 32 and not set(ids).intersection(merged["state_uids"])
        confirmation[scene] = ids
    freeze(ROOT / "protocol.json", {
        "schema": "joint_four_scene_physical_only_v1", "train_scenes": list(SCENES),
        "model_input": "same masked physical entities plus eta; no controller, Flow or scene ID",
        "source_data_sha256": sha(path), "source_counts": counts,
        "models": ["state_concat", "state_residual", "eta_only_wide"],
        "seeds": [17, 23], "optimization_steps": 2000,
        "selection_rule": "existing v3 source-val regret, pairwise, mismatch, NLL rule",
        "development_test": "previously opened phase1_32 panels; diagnosis only, no model changes",
        "fresh_confirmation_state_uids": confirmation,
        "fresh_state_rule": "next 32 eligible untouched TEST states in original sampler order per scene",
        "confirmation_strategy": "reuse exact cache; selected-only Q16 for joint, eta-only, frozen LOSO and same-scene mismatch, plus Safety; no unmeasured full-pool oracle claim",
        "test_model_selection_allowed": False, "new_training_rollouts": 0,
        "resource_limit": "at most 19 of 24 CPUs owned by this task; training requests 2 CPUs and one GPU shard"})
    print(json.dumps({"source_counts": counts, "fresh_confirmation_states": {s: len(v) for s,v in confirmation.items()},
                      "trainval_sha256": sha(path)}), flush=True)


def run():
    os.environ["TRANSFER_Q_WORK_ROOT"] = str(ROOT)
    os.environ["TRANSFER_Q_MAX_STEPS"] = "2000"
    from diagnostics.transferable_state_conditioning_v1 import train_baselines as base
    from diagnostics.transferable_state_conditioning_v1 import train_residual as residual
    from diagnostics.transferable_state_conditioning_v1 import analyze_validation as val
    from diagnostics.transferable_state_conditioning_v1 import analyze_residual_validation as rval
    from diagnostics.transferable_state_conditioning_v1 import choose_final_models as choose
    for module in (base, residual, val, rval, choose):
        module.HERE = ROOT
        module.PANELS = {PANEL: SCENES}
    for kind in ("eta_only_wide", "state_concat"):
        for seed in (17, 23):
            base.fit(PANEL, kind, seed)
    val.evaluate(PANEL)
    for seed in (17, 23):
        residual.fit(PANEL, seed)
    rval.evaluate(PANEL)
    choose.main()
    print("JOINT_FOUR_MODELS_FROZEN", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "fit"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
