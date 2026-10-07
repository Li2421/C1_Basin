"""Source-family validation on a genuinely unseen third Flow controller.

All trained H20 checkpoints saw only canonical and first alternate controllers.
The second alternate Flow and its observed Q16 are used here only for a
diagnostic, never for checkpoint selection.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from . import h20_features as h20
from .rich_intervention_probe import SCENES
from .second_variant import OUT as SECOND
from .train import Critic, OUT, read, write


def run():
    data = dict(np.load(OUT / "rich_probe_dataset_h20.npz"))
    rows = read(OUT / "rich_probe_rows.json")
    second = list(csv.DictReader((SECOND / "intervention_pairs.csv").open()))
    exact = read(SECOND / "pairs.json")
    by_key = {(r["scene"], r["state_uid"], tuple(np.asarray(r["eta"], np.float32))): r
              for r in exact}
    outcomes = {(r["scene"], r["state_uid"], r["eta_uid"]): r for r in second}
    profile = read(SECOND / "protocol.json")["profiles"]
    evaluation = []
    for i, row in enumerate(rows):
        if row["scene"] == "toy_giveway" or row["split"] != "validation":
            continue
        scene, uid = row["scene"], row["state_uid"]
        pair = by_key[(scene, uid, tuple(data["eta"][i]))]
        outcome = outcomes[(scene, uid, pair["eta_uid"])]
        if int(outcome["second_valid"]) != 16:
            continue
        path = h20.rc.OUT / "rich_response_cache" / f"{h20.rc.key(scene, uid, pair['eta'], profile[scene]['alternate_flow_sha256'])}.json"
        context = read(path)
        assert context["valid"], (scene, uid, context.get("error"))
        evaluation.append((i, pair, outcome, np.asarray(context["features"]["mean"], np.float32)))
    assert evaluation
    x = {k: data[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")}
    train_ix = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"], int)
    eta_center = data["eta"][train_ix].mean(0)
    eta_scale = np.maximum(data["eta"][train_ix].std(0), .1)
    eta = (data["eta"] - eta_center) / eta_scale
    records = []
    for kind in ("nominal20", "rich20"):
        raw0 = data["base_c"] if kind.startswith("rich") else data["base_c"][:, :10]
        raw1 = data["alt_c"] if kind.startswith("rich") else data["alt_c"][:, :10]
        fit = np.concatenate((raw0[train_ix], raw1[train_ix]))
        center, scale = fit.mean(0), np.maximum(fit.std(0), .05)
        for seed in (17, 23, 41):
            model = Critic(False)
            empty = model.init(jax.random.PRNGKey(seed), gather(x, [0]),
                               jnp.zeros((1, 3)), jnp.zeros((1, len(center))))
            path = OUT / "rich_probe_models" / kind / f"seed{seed}" / "checkpoint.msgpack"
            params = serialization.from_bytes(empty, path.read_bytes())
            ix = np.asarray([v[0] for v in evaluation], int)
            ctx2 = np.stack([v[3][:len(center)] for v in evaluation])
            inp = np.asarray((ctx2 - center) / scale, np.float32)
            previous = np.asarray((raw1[ix] - center) / scale, np.float32)
            xx, ee = gather(x, data["state_index"][ix]), jnp.asarray(eta[ix])
            logits = np.asarray(model.apply(params, xx, ee, jnp.asarray(inp)))
            wrong = np.asarray(model.apply(params, xx, ee, jnp.asarray(previous)))
            for j, (_, pair, outcome, _) in enumerate(evaluation):
                q = int(outcome["second_success"]) / 16
                records.append({"kind": kind, "seed": seed, "scene": pair["scene"],
                                "state_uid": pair["state_uid"], "eta_uid": pair["eta_uid"],
                                "true_Q16": q, "true_B15": q >= 15/16,
                                "p_correct_context": float(jax.nn.sigmoid(logits[j])),
                                "p_wrong_controller_context": float(jax.nn.sigmoid(wrong[j]))})
    with (SECOND / "unseen_controller_source_predictions.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    aggregate = []
    for kind in ("nominal20", "rich20"):
        for seed in (17, 23, 41):
            for scene in SCENES[1:]:
                selected = [r for r in records if r["kind"] == kind and r["seed"] == seed and r["scene"] == scene]
                if not selected:
                    continue
                y = np.asarray([r["true_Q16"] for r in selected])
                p = np.asarray([r["p_correct_context"] for r in selected])
                w = np.asarray([r["p_wrong_controller_context"] for r in selected])
                nll = float((-y * np.log(np.clip(p, 1e-6, 1-1e-6)) -
                             (1-y) * np.log(np.clip(1-p, 1e-6, 1-1e-6))).mean())
                by = defaultdict(list)
                for r in selected: by[r["state_uid"]].append(r)
                oracle = chosen = wrong_chosen = 0
                for group in by.values():
                    oracle += any(r["true_B15"] for r in group)
                    chosen += max(group, key=lambda r: r["p_correct_context"])["true_B15"]
                    wrong_chosen += max(group, key=lambda r: r["p_wrong_controller_context"])["true_B15"]
                aggregate.append({"kind": kind, "seed": seed, "scene": scene,
                                  "pairs": len(selected), "states": len(by),
                                  "NLL": nll, "MAE": float(np.mean(np.abs(p-y))),
                                  "median_abs_context_response": float(np.median(np.abs(p-w))),
                                  "oracle_B15_states": int(oracle),
                                  "selected_B15_states": int(chosen),
                                  "wrong_context_selected_B15_states": int(wrong_chosen)})
    with (SECOND / "unseen_controller_source_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate[0]))
        writer.writeheader();writer.writerows(aggregate)
    write(SECOND / "unseen_controller_source_audit.json",
          {"seen_controllers_in_training": "canonical + first alternate only",
           "unseen_controller": "second compatible earlier Flow",
           "held_out_source_families": True, "frozen_target_K16_labels_used": False,
           "rows": len(records), "metrics": aggregate})
    print(json.dumps({"pairs": len(evaluation), "metrics": aggregate}))


if __name__ == "__main__":
    run()
