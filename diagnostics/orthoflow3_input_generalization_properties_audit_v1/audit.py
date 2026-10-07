"""Offline input sufficiency/support and frozen critic functional audit.

No new labels, training, model selection, or deployed preprocessing change.
Counterfactual input substitutions are functional diagnostics, not physical
controllers or revised zero-shot results. The target pools were already opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.special import expit
from scipy.stats import spearmanr

from diagnostics.orthoflow3_controller_intervention_generalization_v1.train import Critic, FOLDS, OLD, OUT as PREV
from diagnostics.orthoflow3_controller_context_loso_v1.evaluate import TARGET
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather

OUT = Path(__file__).resolve().parent
KEYS = ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")
NAMES = {
    "agents": ["goal_distance", "velocity_forward", "velocity_lateral", "flow_forward", "flow_lateral",
               "bounded_flow_forward", "bounded_flow_lateral", "radius", "goal_tolerance", "at_goal",
               "policy_axis_required", "policy_origin_forward", "policy_origin_lateral", "policy_axis_forward", "policy_axis_lateral"],
    "pairs": ["relative_forward", "relative_lateral", "distance", "relative_velocity_forward", "relative_velocity_lateral",
              "other_goal_forward", "other_goal_lateral", "clearance", "radius_sum"],
    "obstacles": ["closest_forward", "closest_lateral", "clearance", "normal_forward", "normal_lateral", "curvature",
                  "radius", "extent", "endpoint_a_forward", "endpoint_a_lateral", "endpoint_b_forward", "endpoint_b_lateral", "is_curved"],
    "globals": ["remaining_fraction", "remaining_seconds", "max_seconds", "dt", "max_speed", "flow_committed",
                "wall_margin", "agent_margin", "monitor_active", "progress_window_seconds", "deadlock_hold_seconds",
                "progress_epsilon", "speed_epsilon_fraction", "monitor_elapsed", "monitor_history_empty"],
}


def read(p):
    return json.loads(Path(p).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csvout(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def valid_values(x, key):
    if key == "globals": return x[key]
    am = x["agent_mask"] > 0
    if key == "agents": return x[key][am]
    if key == "pairs":
        mask = am[:, :, None] & am[:, None, :] & ~np.eye(am.shape[1], dtype=bool)[None]
    else:
        mask = am[:, :, None] & (x["obstacle_mask"] > 0)[:, None, :]
    return x[key][mask]


def selected_metrics(z, tt, reference=None):
    robust = np.array([[v is True for v in t["robust"]] for t in tt])
    unk = np.array([[v is None for v in t["robust"]] for t in tt])
    lo = np.array([t["lower"] for t in tt]); hi = np.array([t["upper"] for t in tt])
    z = np.asarray(z, float)
    i = np.arange(len(tt)); j = z.argmax(1)
    p = expit(z[i, j]); yes = robust[i, j]; unknown = unk[i, j]
    result = {"N": len(tt), "B15": int(yes.sum()), "unresolved": int(unknown.sum()),
              "predicted_selected_mean": float(p.mean()), "Q_lower_mean": float(lo[i, j].mean()),
              "Q_upper_mean": float(hi[i, j].mean()), "selected_logit_mean": float(z[i, j].mean()),
              "severe_FP": int(((p > .9) & (hi[i, j] <= .5)).sum()),
              "within_state_logit_std_mean": float(z.std(1).mean())}
    if reference is not None:
        jj = np.asarray(reference).argmax(1); yy = robust[i, jj]; uu = unk[i, jj]
        result.update(top1_changed=int((j != jj).sum()),
                      paired_rescue=int((yes & ~yy & ~(unknown | uu)).sum()),
                      paired_break=int((yy & ~yes & ~(unknown | uu)).sum()))
    return result


def run():
    data = dict(np.load(PREV / "training_dataset.npz")); rows = read(PREV / "training_rows.json")
    frozen = read(PREV / "models_frozen.json"); saved = read(PREV / "target_predictions.json")
    truth = read(OLD / "cached_truth.json")
    write("protocol.json", {"purpose": "post-hoc root-cause audit of already opened frozen targets",
          "new_rollout": 0, "new_training": 0, "model_selection": "none; all three frozen seeds reported",
          "substitutions": ["prior alone", "interaction alone", "curvature channels zero", "source-constant novel channels reset", "context source TRAIN mean"],
          "interpretation": "Functional substitutions may not describe physically valid inputs; not a replacement model or new strict test.",
          "frozen_models_sha256": sha(PREV / "models_frozen.json"),
          "training_dataset_sha256": sha(PREV / "training_dataset.npz")})
    properties, metrics, replay, components = {}, [], [], []
    for fold, target_scene in FOLDS.items():
        train_states = sorted({r["state_index"] for r in rows if r["scene"] != target_scene and r["split"] == "train"})
        source = {k: data[k][train_states] for k in KEYS}
        target = dict(np.load(TARGET / fold / "entities.npz"))
        man = read(TARGET / fold / "manifest.json"); tt = truth[fold]
        assert [m["state_uid"] for m in man] == [t["state_uid"] for t in tt]
        raw_eta = np.asarray([m["eta"] for m in man], np.float32).reshape(-1, 3)
        context = np.load(PREV / f"target_nominal_{fold}.npz")["context"]
        index = np.repeat(np.arange(len(man)), 16)
        novel, neutral = [], {k: v.copy() for k, v in target.items()}
        for key, names in NAMES.items():
            a, b = valid_values(source, key), valid_values(target, key)
            assert len(names) == a.shape[-1]
            for j, name in enumerate(names):
                amin, amax = float(a[:, j].min()), float(a[:, j].max())
                if amax - amin <= 1e-7:
                    changed = np.abs(b[:, j] - amin) > 1e-6
                    if changed.any():
                        novel.append({"tensor": key, "channel": j, "name": name,
                                      "source_constant": amin, "source_zero": abs(amin) < 1e-8,
                                      "target_min": float(b[:, j].min()), "target_max": float(b[:, j].max()),
                                      "target_entity_changed_fraction": float(changed.mean())})
                        neutral[key][..., j] = amin
        curvature = {k: v.copy() for k, v in target.items()}
        curvature["obstacles"][..., [5, 12]] = 0.
        properties[fold] = {"source_train_states": len(train_states), "target_states": len(man),
                            "source_agent_counts": np.unique(source["agent_mask"].sum(-1)).tolist(),
                            "target_agent_counts": np.unique(target["agent_mask"].sum(-1)).tolist(),
                            "source_obstacle_counts": np.unique(source["obstacle_mask"].sum(-1)).tolist(),
                            "target_obstacle_counts": np.unique(target["obstacle_mask"].sum(-1)).tolist(),
                            "source_constant_target_novel": novel}
        for kind in ("structured_full", "contrast_full"):
            for seed in (17, 23, 41):
                folder = PREV / "models" / fold / kind / f"seed{seed}"
                norm = read(folder / "normalization.json"); train = read(folder / "training.json")
                assert sha(folder / "checkpoint.msgpack") == train["checkpoint_sha256"]
                assert target_scene not in norm["source_scenes"] and not norm["target_data_used"]
                eta = (raw_eta - np.array(norm["eta_center"], np.float32)) / np.array(norm["eta_radius"], np.float32)
                c = (context - np.array(norm["context_center"], np.float32)) / np.array(norm["context_scale"], np.float32)
                model = Critic(False)
                template = model.init(jax.random.PRNGKey(0), gather(target, [0]), jnp.zeros((1,3)), jnp.zeros((1,10)))
                params = serialization.from_bytes(template, (folder / "checkpoint.msgpack").read_bytes())

                @jax.jit
                def forward(x, e, cc):
                    z, aux = model.apply(params, x, e, cc,
                                         capture_intermediates=lambda module, method: module.name in ("prior_out", "interaction_out"),
                                         mutable=["intermediates"])
                    v = aux["intermediates"]
                    return z, v["prior_out"]["__call__"][0][...,0], v["interaction_out"]["__call__"][0][...,0]

                def predict(x, cc=None):
                    if cc is None: cc = c
                    parts = [forward(gather(x, index[j:j+128]), jnp.asarray(eta[j:j+128]), jnp.asarray(cc[index[j:j+128]]))
                             for j in range(0,len(index),128)]
                    return [np.concatenate([np.asarray(part[k]) for part in parts]).reshape(len(man),16) for k in range(3)]

                full, prior, interaction = predict(target)
                old = np.asarray(saved["folds"][fold]["logits"][f"{kind}_seed{seed}"])
                err = float(np.max(abs(full-old)))
                assert err < 2e-4, (fold,kind,seed,err)
                assert np.max(abs(full-prior-interaction)) < 2e-5
                replay.append({"fold":fold,"kind":kind,"seed":seed,"logit_replay_max_error":err,
                               "is_source_selected_seed":seed==frozen["folds"][fold][kind]["selected_seed"]})
                context_reset, prior_reset, _ = predict(target, np.zeros_like(c))
                variants = {"frozen_full":full,"prior_only":prior,"interaction_only":interaction,
                            "source_constant_channels_reset":predict(neutral)[0],
                            "context_source_mean":context_reset,"prior_context_source_mean":prior_reset}
                if fold=="ring": variants["curvature_channels_zero"]=predict(curvature)[0]
                for name,z in variants.items():
                    metrics.append({"fold":fold,"kind":kind,"seed":seed,"variant":name,
                                    **selected_metrics(z,tt,full)})
                i = np.arange(len(man)); j = full.argmax(1)
                components.append({"fold":fold,"kind":kind,"seed":seed,
                                   "chosen_prior_logit_mean":float(prior[i,j].mean()),
                                   "chosen_interaction_logit_mean":float(interaction[i,j].mean()),
                                   "prior_candidate_std":float(prior.std(1).mean()),
                                   "interaction_candidate_std":float(interaction.std(1).mean()),
                                   "prior_full_top1_agreement":float((prior.argmax(1)==j).mean()),
                                   "interaction_full_top1_agreement":float((interaction.argmax(1)==j).mean())})
                print(json.dumps({"fold":fold,"kind":kind,"seed":seed,"replay_error":err,
                                  "B15":{k:selected_metrics(v,tt)["B15"] for k,v in variants.items()}}),flush=True)
                del forward
        write("input_support_properties.json", properties)
        csvout("functional_diagnostics.csv",metrics); csvout("replay_checks.csv",replay)
        csvout("branch_contributions.csv",components)
    print("Offline audit complete. NEW ROLLOUT=0.",flush=True)


if __name__ == "__main__":
    run()
