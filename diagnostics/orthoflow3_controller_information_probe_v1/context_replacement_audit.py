"""Matched wrong-controller input intervention on source-family validation states."""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .probe import OUT, Critic, load_scene, read, write


VARIANTS = {
    "controller_eta": (False, False, True),
    "physical_context": (True, True, False),
    "physical_context_plus_id": (True, True, True),
    "fingerprint_eta": (False, True, False),
    "fingerprint_full": (True, True, False),
}


def run():
    results = []
    for scene in ("four_way_intersection", "ring_exchange"):
        data = load_scene(scene)
        rows = data["rows"]
        ix = np.flatnonzero(data["split"] == "validation")
        by_state = {}
        for i in ix:
            by_state.setdefault(rows[i]["state_uid"], []).append(i)
        state_index = data["state_index"]
        xx = gather(data["x"], state_index[ix])
        ee = jnp.asarray(data["eta"][ix])
        for kind, flags in VARIANTS.items():
            for seed in (17, 23, 41):
                dest = OUT / "models" / scene / kind / f"seed{seed}"
                norm = read(dest / "normalization.json")
                if kind.startswith("fingerprint_"):
                    bank = np.asarray(read(OUT / f"fingerprint_bank_{scene}.json")["vectors"], np.float32)
                    center = np.asarray(norm["context_center"], np.float32)
                    scale = np.asarray(norm["context_scale"], np.float32)
                    context = np.broadcast_to(((bank - center) / scale)[:, None, :],
                                              (3, len(rows), 24))
                else:
                    context = data["context"]
                model = Critic(*flags)
                template = model.init(jax.random.PRNGKey(seed),
                                      gather(data["x"], state_index[ix[:1]]),
                                      jnp.zeros((1, 3)), jnp.zeros((1, 24)), jnp.zeros((1, 3)))
                params = serialization.from_bytes(template, (dest / "checkpoint.msgpack").read_bytes())

                def predict(actual, supplied):
                    cc = jnp.asarray(context[supplied, ix])
                    onehot = jnp.broadcast_to(jnp.eye(3)[supplied], (len(ix), 3))
                    return np.asarray(model.apply(params, xx, ee, cc, onehot))

                abs_deltas = []
                changed = eligible = correct_b15 = wrong_b15 = 0
                for actual in range(3):
                    supplied = (actual + 1) % 3
                    good, bad = predict(actual, actual), predict(actual, supplied)
                    abs_deltas.extend(np.abs(jax.nn.sigmoid(good) - jax.nn.sigmoid(bad)).tolist())
                    for pairs in by_state.values():
                        local = [np.where(ix == i)[0][0] for i in pairs]
                        robust = data["success"][actual, pairs] >= 15
                        if not robust.any():
                            continue
                        eligible += 1
                        a, b = int(np.argmax(good[local])), int(np.argmax(bad[local]))
                        changed += int(a != b)
                        correct_b15 += int(robust[a])
                        wrong_b15 += int(robust[b])
                results.append({"scene": scene, "kind": kind, "seed": seed,
                                "median_abs_probability_change": float(np.median(abs_deltas)),
                                "mean_abs_probability_change": float(np.mean(abs_deltas)),
                                "eligible_B15_slots": eligible, "top1_changed": changed,
                                "correct_context_B15": correct_b15,
                                "wrong_context_B15": wrong_b15})
    write(OUT / "context_replacement_audit.json",
          {"scope": "source-family validation; all three controllers seen in training",
           "test_labels_used_for_selection": False, "new_rollout": 0, "results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
