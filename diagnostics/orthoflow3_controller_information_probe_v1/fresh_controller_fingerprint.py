"""Outcome-free v8 Flow response on the already frozen eight-state bank."""
from __future__ import annotations

import json

import numpy as np

from diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_context import RichRuntime
from .fresh_controller import DEST, FLOW
from .probe import OUT, SRC, read, write


def run():
    import jax
    scene = "ring_exchange"
    prior = read(OUT / "fingerprint_bank_ring_exchange.json")
    bank = prior["bank_state_uids"]
    source = read(SRC / "../orthoflow3_loso_partial_count_v1/states.json")
    known = read(SRC / "training_rows.json")
    uid_to_index = {r["state_uid"]: r["state_index"] for r in known if r["scene"] == scene}
    base = RichRuntime(scene)
    controller = RichRuntime(scene, FLOW)
    features = []
    for bank_index, uid in enumerate(bank):
        physical = source[uid_to_index[uid]]["physical"]
        env = base.core.reset(physical)
        values = []
        for root in prior["matched_probe_roots"]:
            key = jax.random.fold_in(jax.random.PRNGKey(root), bank_index)
            raw = np.asarray(controller.alt_flow(env, key), float)
            safe = np.asarray(controller.core.project(env, raw), float)
            goal = np.asarray(env.goals - env.positions, float)
            goal /= np.maximum(np.linalg.norm(goal, axis=1, keepdims=True), 1e-12)
            speed = base.core.cfg.max_speed
            values.append([float(np.mean(np.sum(raw * goal, axis=1)) / speed),
                           float(np.mean(raw[:, 1] * goal[:, 0] - raw[:, 0] * goal[:, 1]) / speed),
                           float(np.mean(np.linalg.norm(safe - raw, axis=1)) / speed)])
        features.extend(np.asarray(values).mean(0).tolist())
    assert len(features) == 24 and np.isfinite(features).all()
    old = np.asarray(prior["vectors"], float)
    write(DEST / "v8_fingerprint.json", {
        "scene": scene, "source_TRAIN_bank_state_uids": bank,
        "checkpoint": str(FLOW), "outcome_labels_read": False,
        "full_task_rollout": False, "vector": features,
        "distance_to_existing_controller_vectors": [float(np.linalg.norm(np.asarray(features) - x))
                                                    for x in old],
    })
    print(json.dumps({"fingerprint_dimension": len(features),
                      "distance_to_existing_controller_vectors":
                      [float(np.linalg.norm(np.asarray(features) - x)) for x in old]}))


if __name__ == "__main__":
    run()
