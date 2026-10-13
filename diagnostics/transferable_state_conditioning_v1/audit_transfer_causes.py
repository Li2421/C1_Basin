"""Posthoc cause audit using frozen checkpoints and already opened outcomes.

No fitting, simulator call or new confirmation claim. Resetting source-constant
feature channels is a sensitivity intervention, not a deployable repair.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.physical_state_critic_formal_v1 import physical_entities
from diagnostics.transferable_state_conditioning_v1.train_residual import StateEtaResidual
from diagnostics.transferable_state_conditioning_v1.assemble_source import ETA_CENTER, ETA_RADIUS

HERE = Path(__file__).resolve().parent
STAGE = HERE / "stage_v3_coverage"
FINAL = STAGE / "final_phase1_32"
FORMAL = HERE.parent / "physical_state_critic_formal_v1"
OUT = HERE / "posthoc_cause_audit"
SCENES = ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if min(a.std(), b.std()) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap(v):
    v = np.asarray(v, float)
    rng = np.random.default_rng(20261009)
    b = v[rng.integers(len(v), size=(20000, len(v)))].mean(1)
    return {"difference": float(v.mean()), "ci95": np.quantile(b, [.025, .975]).tolist(),
            "physical_states": len(v)}


def source_summary(d):
    result = {}
    for scene in sorted(set(d["scene"])):
        result[scene] = {}
        for split in ("train", "validation"):
            mask = (d["scene"] == scene) & (d["split"] == split)
            k, f, u = (d[key][mask] for key in ("success", "failure", "numerical"))
            assert np.all(k+f+u == 4)
            q = k/4
            result[scene][split] = {
                "states": int(mask.sum()), "pairs": int(k.size), "seeds": int(4*k.size),
                "mean_Q_lower": float(q.mean()), "mean_Q_upper": float(((k+u)/4).mean()),
                "all_16_candidates_4of4_states": int(np.sum((k == 4).all(1))),
                "mean_full_success_candidates": float((k == 4).sum(1).mean()),
                "nonconstant_Q_states": int(np.sum(np.ptp(q, axis=1) > 0)),
                "numerical_seeds": int(u.sum()),
            }
    return result


def active_columns(x, key):
    am = x["agent_mask"].astype(bool)
    if key == "agents":
        mask = am
    elif key == "pairs":
        n = am.shape[1]
        mask = am[:, :, None] & am[:, None, :] & ~np.eye(n, dtype=bool)[None]
    elif key == "obstacles":
        mask = am[:, :, None] & x["obstacle_mask"][:, None, :].astype(bool)
    else:
        raise ValueError(key)
    return x[key][mask]


def reset_unseen_constants(source_x, target_x):
    changed = {k: v.copy() for k, v in target_x.items()}
    audit = {}
    for key in ("agents", "pairs", "obstacles"):
        s, t = active_columns(source_x, key), active_columns(target_x, key)
        constant = np.ptp(s, axis=0) < 1e-7
        novel = np.any(np.abs(t-s[0]) > 1e-6, axis=0)
        columns = np.flatnonzero(constant & novel)
        audit[key] = [{"index": int(j), "source_constant": float(s[0, j]),
                       "target_min": float(t[:, j].min()), "target_max": float(t[:, j].max())}
                      for j in columns]
        for j in columns:
            changed[key][..., j] = s[0, j]
    audit["agent_counts_source"] = sorted(set(source_x["agent_mask"].sum(1).tolist()))
    audit["agent_counts_target"] = sorted(set(target_x["agent_mask"].sum(1).tolist()))
    audit["obstacle_counts_source"] = sorted(set(source_x["obstacle_mask"].sum(1).tolist()))
    audit["obstacle_counts_target"] = sorted(set(target_x["obstacle_mask"].sum(1).tolist()))
    return changed, audit


class FrozenModel:
    def __init__(self, info, x, normalized_eta):
        self.info = info
        self.residual = info["state_kind"] == "state_residual"
        self.net = StateEtaResidual() if self.residual else rep.Critic()
        ex = {k: jnp.asarray(v[:1]) for k, v in x.items()}
        ee = jnp.asarray(normalized_eta[:1] if self.residual else normalized_eta[:1, 0])
        self.initial = self.net.init(jax.random.PRNGKey(info["state_seed"]), ex, ee)
        ck = Path(info["state_checkpoint"])
        assert sha(ck) == info["state_checkpoint_sha256"]
        self.params = serialization.from_bytes(self.initial, ck.read_bytes())
        self.apply = jax.jit(lambda xx, eta: self.net.apply(self.params, xx, eta))

    def predict(self, x, eta, prior=None):
        eta = np.asarray((eta-ETA_CENTER)/ETA_RADIUS, np.float32)
        rows = []
        if self.residual:
            for a in range(0, len(eta), 32):
                rows.append(np.asarray(self.apply({k: jnp.asarray(v[a:a+32]) for k, v in x.items()},
                                                  jnp.asarray(eta[a:a+32]))))
            z = np.concatenate(rows)
            assert prior is not None
            return z + prior
        ii = np.repeat(np.arange(len(eta)), 16)
        ee = eta.reshape(-1, 3)
        for a in range(0, len(ii), 256):
            rows.append(np.asarray(self.apply({k: jnp.asarray(v[ii[a:a+256]]) for k, v in x.items()},
                                              jnp.asarray(ee[a:a+256]))))
        return np.concatenate(rows).reshape(len(eta), 16)

    def zero_channel_weights(self, channels):
        p = self.params["params"]["physical_encoder"]["obstacle1"]["kernel"]
        p0 = self.initial["params"]["physical_encoder"]["obstacle1"]["kernel"]
        out = []
        for row in channels:
            if row["source_constant"] != 0:
                continue
            j = row["index"]
            a, b = np.asarray(p0[j], float), np.asarray(p[j], float)
            scale = float(np.dot(a, b)/np.dot(a, a))
            out.append({"column": j, "learned_vs_initial_cosine": float(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)),
                        "weight_decay_scale": scale,
                        "max_residual_after_scalar_decay": float(np.max(np.abs(b-scale*a)))})
        return out


def anchor_metrics(z, values):
    q = values[:, :, 8:, 0].mean(2)
    q_upper = (values[:, :, 8:, 0]+values[:, :, 8:, 4]).mean(2)
    choice = z.argmax(1)
    row = np.arange(len(q))
    selected = q[row, choice]
    # Averaging every other state's choice is the exact expectation of a
    # uniform same-scene derangement on a common candidate bank.
    all_other = q[:, choice]
    mismatched = (all_other.sum(1)-np.diag(all_other))/(len(q)-1)
    pred = 1/(1+np.exp(-np.clip(z, -50, 50)))
    interaction = z-z.mean(0, keepdims=True)-z.mean(1, keepdims=True)+z.mean()
    q_interaction = q-q.mean(0, keepdims=True)-q.mean(1, keepdims=True)+q.mean()
    return {"selected_counts": np.bincount(choice, minlength=16).tolist(),
            "mean_success_lower": float(selected.mean()),
            "mean_success_upper": float(q_upper[row, choice].mean()),
            "matched_minus_mismatch": bootstrap(selected-mismatched),
            "state_effect_corr": corr(pred-pred.mean(0), q-q.mean(0)),
            "double_centered_interaction_corr": corr(interaction, q_interaction),
            "logit_interaction_rms": float(np.sqrt(np.mean(interaction**2))),
            "eta_main_effect_rms": float(np.std(z.mean(0))),
            "posthoc_fixed_success": float(q.mean(0).max())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = read(STAGE / "final_models_frozen.json")
    split = read(FINAL / "frozen_splits.json")
    result = {"schema": "posthoc_transfer_cause_audit_v1", "new_rollouts": 0,
              "new_training": False, "final_states_now_used_for_scientific_diagnosis": True,
              "sensitivity_not_new_confirmatory_method": True, "scenes": {}}
    fixed = np.asarray(split["common_anchor_eta"], np.float32)
    for scene in SCENES:
        panel = "loso_"+scene
        dpath = STAGE / "source_data" / panel / "trainval.npz"
        with np.load(dpath) as fd:
            d = dict(fd)
        train = d["split"] == "train"
        source_x = {k[7:]: v[train] for k, v in d.items() if k.startswith("entity_")}
        proposals = {r["state_uid"]: r for r in read(FORMAL / "loso" / scene / "test" / "proposals.json")["states"]}
        source_rows = {r["alias"]: r for r in read(FORMAL / f"states_{scene}.json")["states"]}
        ids = split["final_confirmation"][scene]["state_uids"]
        x = physical_entities.batch([{"scene": scene, "physical": source_rows[proposals[u]["state_alias"]]["physical"]} for u in ids])
        reset_x, feature_audit = reset_unseen_constants(source_x, x)
        info = frozen["panels"][panel]
        model = FrozenModel(info, x, np.broadcast_to((fixed-ETA_CENTER)/ETA_RADIUS, (len(ids), 16, 3)))
        entry = {"source_data_sha256": sha(dpath), "source_labels": source_summary(d),
                 "feature_support": feature_audit,
                 "zero_source_obstacle_weight_audit": model.zero_channel_weights(feature_audit["obstacles"]),
                 "banks": {}}
        for bank in ("anchor", "proposal"):
            predpath = FINAL / "predictions" / panel / f"{bank}.npz"
            pred = np.load(predpath)
            assert ids == pred["state_uids"].tolist()
            eta = pred["eta"]
            original = model.predict(x, eta, pred["eta_logits"])
            assert np.allclose(original, pred["state_logits"], rtol=2e-5, atol=5e-5)
            reset = model.predict(reset_x, eta, pred["eta_logits"])
            outcome = np.load(FINAL / "final" / bank / scene / "outcomes.npz")
            assert outcome["state_uids"].tolist() == ids
            values = outcome["values"][:, :16]
            q = values[..., 0].mean(2)
            q_upper = (values[..., 0]+values[..., 4]).mean(2)
            bank_result = {"original_prediction_sha256": sha(predpath), "variants": {}}
            for name, z in (("original", original), ("source_constant_features_reset", reset)):
                choice = z.argmax(1)
                selected = q[np.arange(len(ids)), choice]
                upper = q_upper[np.arange(len(ids)), choice]
                m = {"B15_confirmed": int(np.sum(selected >= 15/16)),
                     "B15_possible": int(np.sum(upper >= 15/16)),
                     "mean_success_lower": float(selected.mean()),
                     "mean_success_upper": float(upper.mean()),
                     "selection_change_fraction": float(np.mean(choice != original.argmax(1))),
                     "logit_change_rms": float(np.sqrt(np.mean((z-original)**2)))}
                if bank == "anchor":
                    m["independent_evaluation_half"] = anchor_metrics(z, values)
                bank_result["variants"][name] = m
            entry["banks"][bank] = bank_result
            np.savez_compressed(OUT / f"{scene}_{bank}.npz", state_uids=np.asarray(ids),
                                eta=eta, original_logits=original, reset_logits=reset,
                                reset_choice=reset.argmax(1))
        result["scenes"][scene] = entry
        (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(scene, json.dumps({"novel_constant_features": feature_audit,
                                "banks": entry["banks"]}), flush=True)


if __name__ == "__main__":
    main()
