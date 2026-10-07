"""Matched before/after Ring matrix-support comparison on the same held panels."""
from __future__ import annotations

import csv
from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .data_geometry_audit import certified_sign
from .probe import OUT, Critic, load_scene, observed_nll, read


def evaluate(z, data, controller, split="validation"):
    ix = np.flatnonzero(data["split"] == split)
    by_state = {}
    for i in ix:
        by_state.setdefault(data["rows"][i]["state_uid"], []).append(i)
    available = chosen_b15 = unknown = 0
    selected = {}
    for uid, pairs in by_state.items():
        robust = data["success"][controller, pairs] >= 15
        if not robust.any():
            continue
        available += 1
        pick = pairs[int(np.argmax(z[controller, pairs]))]
        selected[uid] = pick
        chosen_b15 += int(data["success"][controller, pick] >= 15)
        unknown += int(data["success"][controller, pick] < 15 and
                       data["failure"][controller, pick] < 2 and
                       data["success"][controller, pick] + data["failure"][controller, pick] < 16)
    reversal_total = reversal_correct = 0
    for pairs in by_state.values():
        for a, b in combinations(pairs, 2):
            for other in (0, 1):
                t0 = certified_sign(data, other, a, b)
                tc = certified_sign(data, controller, a, b)
                if t0 * tc < 0:
                    reversal_total += 1
                    reversal_correct += int(np.sign(z[other, a] - z[other, b]) == t0 and
                                            np.sign(z[controller, a] - z[controller, b]) == tc)
    return {"available_B15": available, "selected_B15": chosen_b15,
            "selected_unknown": unknown, "controller_reversals": reversal_total,
            "controller_reversals_correct": reversal_correct, "selected": selected,
            "NLL": observed_nll(z[controller, ix], data["success"][controller, ix],
                                data["failure"][controller, ix])}


def predict_all(directory, data, kind, seed):
    model = Critic(kind == "fingerprint_full", True, False)
    x = data["x"]
    si = data["state_index"]
    template = model.init(jax.random.PRNGKey(seed), gather(x, si[:1]),
                          jnp.zeros((1, 3)), jnp.zeros((1, 24)), jnp.zeros((1, 3)))
    params = serialization.from_bytes(template, (directory / "checkpoint.msgpack").read_bytes())
    norm = read(directory / "normalization.json")
    bank = np.asarray(read(OUT / "fingerprint_bank_ring_exchange.json")["vectors"], np.float32)
    center = np.asarray(norm["context_center"], np.float32)
    scale = np.asarray(norm["context_scale"], np.float32)
    result = []
    for controller in range(3):
        context = jnp.broadcast_to(jnp.asarray((bank[controller] - center) / scale),
                                   (len(si), 24))
        identity = jnp.broadcast_to(jnp.eye(3)[controller], (len(si), 3))
        result.append(np.asarray(model.apply(params, gather(x, si), jnp.asarray(data["eta"]),
                                             context, identity)))
    return np.stack(result)


def run():
    data = load_scene("ring_exchange")
    rows = []
    for kind in ("fingerprint_eta", "fingerprint_full"):
        for seed in (17, 23, 41):
            dest = OUT / "models/ring_exchange"
            old_dir = dest / (kind + "_heldout_controller2") / f"seed{seed}"
            new_dir = dest / (kind + "_heldout_controller2_crossmatrix") / f"seed{seed}"
            old_z = np.load(old_dir / "validation_predictions.npz")["logits"]
            new_z = np.load(new_dir / "validation_predictions.npz")["logits"][:, :len(data["rows"])]
            old = evaluate(old_z, data, 2)
            new = evaluate(new_z, data, 2)
            old_train = evaluate(predict_all(old_dir, data, kind, seed), data, 2, "train")
            new_train = evaluate(predict_all(new_dir, data, kind, seed), data, 2, "train")
            rescue = break_count = 0
            for uid in old["selected"]:
                a = old["selected"][uid]
                b = new["selected"][uid]
                rescue += int(data["success"][2, a] < 15 and data["success"][2, b] >= 15)
                break_count += int(data["success"][2, a] >= 15 and data["success"][2, b] < 15)
            ix = np.flatnonzero(data["split"] == "validation")
            rows.append({"kind": kind, "seed": seed,
                         "source_controller_VAL_NLL_old": observed_nll(old_z[:2, ix],
                             data["success"][:2, ix], data["failure"][:2, ix]),
                         "source_controller_VAL_NLL_new": observed_nll(new_z[:2, ix],
                             data["success"][:2, ix], data["failure"][:2, ix]),
                         "held_controller_VAL_NLL_old": old["NLL"],
                         "held_controller_VAL_NLL_new": new["NLL"],
                         "available_B15": old["available_B15"],
                         "held_controller_B15_old": old["selected_B15"],
                         "held_controller_B15_new": new["selected_B15"],
                         "held_controller_unknown_old": old["selected_unknown"],
                         "held_controller_unknown_new": new["selected_unknown"],
                         "held_controller_reversals": old["controller_reversals"],
                         "reversals_correct_old": old["controller_reversals_correct"],
                         "reversals_correct_new": new["controller_reversals_correct"],
                         "held_controller_TRAIN_states_available_B15": old_train["available_B15"],
                         "held_controller_TRAIN_states_B15_old": old_train["selected_B15"],
                         "held_controller_TRAIN_states_B15_new": new_train["selected_B15"],
                         "held_controller_TRAIN_states_reversals": old_train["controller_reversals"],
                         "held_controller_TRAIN_states_reversals_correct_old": old_train["controller_reversals_correct"],
                         "held_controller_TRAIN_states_reversals_correct_new": new_train["controller_reversals_correct"],
                         "held_controller_TRAIN_states_NLL_old": old_train["NLL"],
                         "held_controller_TRAIN_states_NLL_new": new_train["NLL"],
                         "paired_rescue": rescue, "paired_break": break_count,
                         "train_states_old": read(old_dir / "summary.json")["train_states"],
                         "train_states_new": read(new_dir / "summary.json")["train_states"]})
    path = OUT / "crossmatrix_ring_v2/old_vs_crossmatrix.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} matched models -> {path}")


if __name__ == "__main__":
    run()
