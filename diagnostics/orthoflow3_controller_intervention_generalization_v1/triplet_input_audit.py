"""Held-out source-family input-use audit for the three-controller probe."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .rich_intervention_probe import SCENES
from .second_variant import OUT as SECOND
from .train import Critic, OUT, read, write
from .train_triplet_probe import make_model


def run(kind="full"):
    data = dict(np.load(SECOND / "triplet_dataset_h20.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    x = {k: data[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")}
    train_ix = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"], int)
    eta_center = data["eta"][train_ix].mean(0)
    eta_scale = np.maximum(data["eta"][train_ix].std(0), .1)
    eta = (data["eta"] - eta_center) / eta_scale
    width = 10 if kind == "additive" else 24
    fit = np.concatenate([data["base_c"][train_ix, :width], data["alt_c"][train_ix, :width],
                          data["second_c"][train_ix][data["second_full"][train_ix], :width]])
    center, scale = fit.mean(0), np.maximum(fit.std(0), .05)
    contexts = [(data[k][:, :width] - center) / scale for k in ("base_c", "alt_c", "second_c")]
    s = [data[k] for k in ("base_s", "alt_s", "second_s")]
    f = [data[k] for k in ("base_f", "alt_f", "second_f")]
    records, metrics = [], []
    for seed in (17, 23, 41):
        model = make_model(kind)
        empty = model.init(jax.random.PRNGKey(seed), gather(x, [0]), jnp.zeros((1, 3)), jnp.zeros((1, width)))
        folder = {"full": "triplet_models", "additive": "triplet_nominal_additive_models",
                  "bounded1": "triplet_bounded1_models", "bounded3": "triplet_bounded3_models",
                  "bounded3_interaction": "triplet_bounded3_interaction_models"}[kind]
        params = serialization.from_bytes(empty, (SECOND / folder / f"seed{seed}" / "checkpoint.msgpack").read_bytes())
        for scene in SCENES:
            ix = np.asarray([i for i, r in enumerate(rows) if r["scene"] == scene and r["split"] == "validation"], int)
            for j in range(3):
                active = ix[(s[j][ix] + f[j][ix]) > 0]
                if not len(active):
                    continue
                wrong_j = (j + 1) % (2 if scene == "toy_giveway" else 3)
                physical = gather(x, data["state_index"][active])
                actual = np.asarray(jax.nn.sigmoid(model.apply(params, physical, jnp.asarray(eta[active]),
                                                             jnp.asarray(contexts[j][active]))))
                wrong = np.asarray(jax.nn.sigmoid(model.apply(params, physical, jnp.asarray(eta[active]),
                                                            jnp.asarray(contexts[wrong_j][active]))))
                state_ix = data["state_index"][active].copy()
                unique = sorted(set(state_ix))
                rotated = {u: unique[(n + 1) % len(unique)] for n, u in enumerate(unique)}
                shuffled_state = np.asarray([rotated[u] for u in state_ix], int)
                state_prob = np.asarray(jax.nn.sigmoid(model.apply(params, gather(x, shuffled_state),
                                                                  jnp.asarray(eta[active]),
                                                                  jnp.asarray(contexts[j][active]))))
                groups = defaultdict(list)
                for n, idx in enumerate(active):
                    groups[rows[idx]["state_uid"]].append(n)
                    records.append({"seed": seed, "scene": scene, "controller": j,
                                    "state_uid": rows[idx]["state_uid"], "pair_index": int(idx),
                                    "observed_success": int(s[j][idx]), "observed_failure": int(f[j][idx]),
                                    "p_correct": float(actual[n]), "p_wrong_controller": float(wrong[n]),
                                    "p_wrong_state": float(state_prob[n])})
                selected, wrong_selected, state_selected, oracle = [], [], [], []
                for indices in groups.values():
                    ixg = np.asarray(indices, int)
                    good = (s[j][active[ixg]] + f[j][active[ixg]] == 16) & (f[j][active[ixg]] <= 1)
                    oracle.append(bool(good.any()))
                    selected.append(bool(good[np.argmax(actual[ixg])]))
                    wrong_selected.append(bool(good[np.argmax(wrong[ixg])]))
                    state_selected.append(bool(good[np.argmax(state_prob[ixg])]))
                metrics.append({"seed": seed, "scene": scene, "controller": j,
                                "states": len(groups), "oracle_B15": int(sum(oracle)),
                                "correct_context_B15": int(sum(selected)),
                                "wrong_controller_context_B15": int(sum(wrong_selected)),
                                "wrong_state_B15": int(sum(state_selected)),
                                "median_abs_context_score_change": float(np.median(np.abs(actual-wrong))),
                                "top1_change_wrong_context": int(sum(np.argmax(actual[v]) != np.argmax(wrong[v]) for v in groups.values()))})
    prefix = {"full": "triplet_input", "additive": "triplet_nominal_additive_input",
              "bounded1": "triplet_bounded1_input", "bounded3": "triplet_bounded3_input",
              "bounded3_interaction": "triplet_bounded3_interaction_input"}[kind]
    with (SECOND / f"{prefix}_predictions.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    with (SECOND / f"{prefix}_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics[0]))
        writer.writeheader();writer.writerows(metrics)
    write(SECOND / f"{prefix}_audit.json",
          {"kind": kind, "heldout_source_families": True, "target_labels_used": False,
           "matched_wrong_controller": True, "state_shuffle_within_scene": True,
           "metrics": metrics})
    print(json.dumps(metrics))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("full", "additive", "bounded1", "bounded3", "bounded3_interaction"), default="full")
    run(parser.parse_args().kind)
