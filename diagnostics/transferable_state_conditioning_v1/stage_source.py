"""Source-only pilot from exact cached Q4 rows; no target outcomes.

The state index ranges are fixed independently of rollout labels. This pilot
is DEVELOPMENT and never substitutes for untouched final confirmation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from diagnostics.physical_state_critic_formal_v1 import physical_entities
from diagnostics.transferable_state_conditioning_v1.assemble_source import (
    HERE, FORMAL, PANELS, ETA_CENTER, ETA_RADIUS, read, sha,
)
from shared_rollout_db.src.rollout_db import connect

STAGE = Path(os.environ.get("TRANSFER_Q_WORK_ROOT", str(HERE / "stage_v1")))
INDEX_SPLITS = {
    "toy_give_way": {"train": list(range(0, 128)), "validation": list(range(128, 144))},
    "double_bottleneck": {"train": list(range(0, 48)), "validation": list(range(48, 64))},
    "four_way_intersection": {"train": list(range(0, 48)), "validation": list(range(48, 64))},
    "ring_exchange": {"train": list(range(56, 152)), "validation": list(range(152, 168))},
}
if os.environ.get("TRANSFER_Q_STAGE_PROFILE") == "coverage":
    INDEX_SPLITS = {
        **INDEX_SPLITS,
        "toy_give_way": {"train": list(range(0, 256)), "validation": list(range(256, 288))},
        "ring_exchange": {"train": list(range(56, 168)), "validation": list(range(168, 184))},
    }


def freeze_selection():
    path = STAGE / "source_index_splits.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "source_only_development_stage_v1",
              "selection": INDEX_SPLITS,
              "rule": "contiguous prechosen index ranges; no success-conditioned filtering",
              "target_labels_used": False}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists():
        assert path.read_text() == payload
    else:
        path.write_text(payload)
    return path


def scene_rows(scene, indices=None):
    panel = FORMAL / "loso" / scene / "train"
    cases = read(panel / "cases.json")
    proposals = read(panel / "proposals.json")["states"]
    states = {r["alias"]: r for r in read(FORMAL / f"states_{scene}.json")["states"]}
    indices = INDEX_SPLITS[scene] if indices is None else indices
    rows = []
    with connect(True) as db:
        for split in ("train", "validation"):
            for i in indices[split]:
                p = proposals[i]
                state_cases = cases[i*64:(i+1)*64]
                assert len(state_cases) == 64
                assert all(c["state_index"] == i and c["state_uid"] == p["state_uid"]
                           for c in state_cases)
                success = np.zeros((16, 4), np.int8)
                failure = np.zeros((16, 4), np.int8)
                numerical = np.zeros((16, 4), np.int8)
                for c in state_cases:
                    path = panel / "raw" / f"{c['rollout_uid']}.json"
                    if path.exists():
                        r = read(path)
                        assert r["case"] == c
                        values = [r["success"], r["numerical_failure"]]
                    else:
                        r = db.execute("SELECT success,numerical_failure,"
                                       "state_uid,eta_uid,controller_uid,seed_key "
                                       "FROM rollout WHERE rollout_uid=? "
                                       "AND conflict_quarantined=0", (c["rollout_uid"],)).fetchone()
                        assert r is not None, f"missing exact source seed: {scene} state {i}"
                        assert (r["state_uid"], r["eta_uid"], r["controller_uid"], r["seed_key"]) == (
                            c["state_uid"], c["eta_uid"], c["controller_uid"], c["seed_key"])
                        values = [r["success"], r["numerical_failure"]]
                    j, k = c["eta_index"], c["future_index"]
                    success[j, k], numerical[j, k] = values
                    failure[j, k] = 1 - success[j, k] - numerical[j, k]
                    assert failure[j, k] >= 0
                physical = states[p["state_alias"]]["physical"]
                rows.append({"scene": scene, "split": split, "state_uid": p["state_uid"],
                             "state_index": i, "physical": physical,
                             "eta": np.asarray(p["eta"], np.float32),
                             "success": success.sum(1), "failure": failure.sum(1),
                             "numerical": numerical.sum(1)})
    return rows


def assemble(panel):
    assert panel in PANELS
    split_path = freeze_selection()
    output = STAGE / "source_data" / panel
    output.mkdir(parents=True, exist_ok=True)
    target = output / "trainval.npz"
    if target.exists():
        manifest = read(output / "manifest.json")
        assert manifest["trainval_sha256"] == sha(target)
        return manifest
    rows = [r for scene in PANELS[panel] for r in scene_rows(scene)]
    assert len({r["state_uid"] for r in rows}) == len(rows)
    entities = physical_entities.batch(
        [{"scene": r["scene"], "physical": r["physical"]} for r in rows])
    assert np.all(entities["agents"][..., 3:7] == 0)
    assert np.all(entities["agents"][..., 10:15] == 0)
    assert np.all(entities["globals"] == 0)
    eta = np.stack([r["eta"] for r in rows])
    data = {"eta": (eta - ETA_CENTER) / ETA_RADIUS,
            "eta_raw": eta,
            "success": np.stack([r["success"] for r in rows]),
            "failure": np.stack([r["failure"] for r in rows]),
            "numerical": np.stack([r["numerical"] for r in rows]),
            "state_uids": np.asarray([r["state_uid"] for r in rows]),
            "state_indices": np.asarray([r["state_index"] for r in rows]),
            "scene": np.asarray([r["scene"] for r in rows]),
            "split": np.asarray([r["split"] for r in rows]),
            **{f"entity_{k}": v for k, v in entities.items()}}
    np.savez_compressed(target, **data)
    manifest = {"schema": "source_only_development_stage_trainval_v1",
                "panel": panel, "source_scenes": list(PANELS[panel]),
                "train_states": int(np.sum(data["split"] == "train")),
                "validation_states": int(np.sum(data["split"] == "validation")),
                "trainval_sha256": sha(target),
                "source_index_splits_sha256": sha(split_path),
                "feature_code_sha256": sha(physical_entities.__file__),
                "test_labels_opened": False,
                "cache_policy": "exact raw rollout UID first, then exact conflict-free DB row",
                "rollout_seeds_per_state_eta": 4}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", choices=tuple(PANELS))
    a = p.parse_args()
    if a.panel:
        print(json.dumps(assemble(a.panel), indent=2))
    else:
        print(freeze_selection())
