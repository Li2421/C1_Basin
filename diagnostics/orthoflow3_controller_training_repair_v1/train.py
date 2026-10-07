"""Matched source training and independent-family gate, without target labels.

Full interaction keeps the existing physical-context architecture and pure
observed-count NLL. No artificial controller IDs, ranking loss or generator edits.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
import flax.linen as nn

from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic, read, write
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from .data import OUT, CONTROLLERS, digest, old

jax.config.update("jax_enable_x64", False)

SEEDS = (17, 23, 41)
KINDS = ("eta_only", "additive", "physical_context")
STATE_NORMALIZED = "physical_context_state_normalized"


class Additive(nn.Module):
    @nn.compact
    def __call__(self, x, eta, context, ids):
        h = rep.Encoder(name="physical_encoder")(x)
        c = nn.silu(nn.Dense(32, name="context_encoder")(context))
        # c(h,eta) is eta-conditioned: using it here would allow interaction
        # indirectly! Only the eta-independent nominal first 10 features enter A.
        # The caller zeros all candidate-response channels for this baseline.
        a = nn.silu(nn.Dense(128, name="state1")(jnp.concatenate((h, c), -1)))
        a = nn.silu(nn.Dense(64, name="state2")(a))
        e = nn.silu(nn.Dense(32, name="eta_encoder")(eta))
        e = nn.silu(nn.Dense(128, name="eta1")(e))
        e = nn.silu(nn.Dense(64, name="eta2")(e))
        return (nn.Dense(1, name="state_out")(a)+nn.Dense(1, name="eta_out")(e))[..., 0]


def model_for(kind):
    return Additive() if kind == "additive" else Critic(kind != "eta_only", kind != "eta_only", False)


def controller_balanced_nll(z, s, f):
    terms = s*jax.nn.softplus(-z)+f*jax.nn.softplus(z)
    return jnp.mean(jnp.sum(terms, axis=1)/jnp.maximum(1., jnp.sum(s+f, axis=1)))


def load():
    assert read(OUT / "controller_path_audit.json")["passed"]
    assert read(OUT / "alignment_audit_all.json")["all_checks_passed"]
    d = dict(np.load(OUT / "dataset.npz"))
    d["x"] = dict(np.load(OUT / "entities.npz"))
    d["rows"] = read(OUT / "pairs.json")
    if not np.isfinite(d["context"]).all():
        raise RuntimeError("A physical response probe failed: audit it; do not impute a zero context")
    tr = d["split"] == "train"
    # Only source TRAIN data enters either normalization.
    ee = d["eta"][tr]
    cc = d["context"][:, tr][d["valid"][:, tr]]
    norm = {"eta_center": ee.mean(0), "eta_scale": np.maximum(ee.std(0), .1),
            "context_center": cc.mean(0), "context_scale": np.maximum(cc.std(0), .05)}
    d["eta"] = ((d["eta"]-norm["eta_center"])/norm["eta_scale"]).astype(np.float32)
    d["context"] = ((d["context"]-norm["context_center"])/norm["context_scale"]).astype(np.float32)
    # Missing/numerical physical contexts contribute no likelihood, not fake negatives.
    d["context"] = np.nan_to_num(d["context"])
    for key in ("success", "failure"):
        d[key] = np.where(d["valid"], d[key], 0.)
    d["norm"] = {k: v.tolist() for k, v in norm.items()}
    assert np.isfinite(d["eta"]).all() and np.isfinite(d["context"]).all()
    return d


def metrics(z, d, indices):
    indices = np.asarray(indices)
    s, f = d["success"][:, indices], d["failure"][:, indices]
    p = np.asarray(jax.nn.sigmoid(z[:, indices]))
    n = s+f
    active = n > 0
    q = s/np.maximum(n, 1.)
    result = {"NLL": float(controller_balanced_nll(jnp.asarray(z[:, indices]), jnp.asarray(s), jnp.asarray(f))),
              "MAE": float(np.abs(p-q)[active].mean())}
    ss, ff = d["standard_success"], d["standard_failure"]
    lo, hi = ss/16., (16.-ff)/16.
    groups = {}
    for i in indices:
        groups.setdefault(d["rows"][i]["state_uid"], []).append(i)
    eligible = correct = unknown = severe = 0
    q_selected, q_oracle, predictions = [], [], []
    picks = []
    for c in range(3):
        for suid, ii in groups.items():
            good = ss[c, ii] >= 15
            if not good.any():
                continue
            eligible += 1
            j = ii[int(np.argmax(z[c, ii]))]
            correct += int(ss[c, j] >= 15)
            unknown += int(ss[c, j] < 15 and ff[c, j] < 2)
            severe += int(z[c, j] > np.log(9) and hi[c, j] <= .5)
            n_all = d["success"][c] + d["failure"][c]
            q_all = d["success"][c] / np.maximum(n_all, 1.)
            q_selected.append(float(q_all[j]))
            q_oracle.append(float(np.max(q_all[ii])))
            predictions.append(float(jax.nn.sigmoid(z[c, j])))
            picks.append({"controller": c, "state_uid": suid, "index": int(j),
                          "B15": bool(ss[c, j] >= 15), "Q_lower": float(lo[c, j]), "Q_upper": float(hi[c, j])})
    result.update(eligible=eligible, selected_B15=correct, selected_unknown=unknown,
                  severe_false_positive=severe, picks=picks,
                  selected_observed_Q=float(np.mean(q_selected)) if q_selected else None,
                  oracle_observed_Q=float(np.mean(q_oracle)) if q_oracle else None,
                  observed_Q_regret=float(np.mean(np.asarray(q_oracle)-q_selected)) if q_selected else None,
                  selected_predicted_probability=float(np.mean(predictions)) if predictions else None,
                  observed_Q_caveat="Uses measured successes/trials; numerical outcomes not imputed. B15 uses standard-seed bounds.")
    # Strong pair orderings use Q16 intervals: never impute missing outcomes.
    def sign(c, i, j):
        return 1 if lo[c, i]-hi[c, j] >= .25 else -1 if hi[c, i]-lo[c, j] <= -.25 else 0
    total = right = 0
    families = set()
    for suid, ii in groups.items():
        for i, j in combinations(ii, 2):
            for a, b in combinations(range(3), 2):
                sa, sb = sign(a, i, j), sign(b, i, j)
                if sa*sb >= 0:
                    continue
                total += 1
                families.add(suid)
                right += int(np.sign(z[a, i]-z[a, j]) == sa and np.sign(z[b, i]-z[b, j]) == sb)
    result["controller_reversals"] = {"total": total, "correct": right, "families": len(families),
                                      "accuracy": right/total if total else None}
    total = right = 0
    lists = list(groups.values())
    for a, b in combinations(lists, 2):
        assert [d["rows"][i]["eta_uid"] for i in a] == [d["rows"][i]["eta_uid"] for i in b]
        for c in range(3):
            for i, j in combinations(range(len(a)), 2):
                sa, sb = sign(c, a[i], a[j]), sign(c, b[i], b[j])
                if sa*sb >= 0:
                    continue
                total += 1
                right += int(np.sign(z[c, a[i]]-z[c, a[j]]) == sa and np.sign(z[c, b[i]]-z[c, b[j]]) == sb)
    result["state_reversals"] = {"total": total, "correct": right, "accuracy": right/total if total else None}
    return result


def train(kind, seed):
    dest = OUT / "models" / kind / f"seed{seed}"
    if (dest / "summary.json").exists():
        return
    d = load()
    state_normalization = None
    if kind == STATE_NORMALIZED:
        from .state_normalization import normalize_entities
        d["x"], state_normalization = normalize_entities(d["x"], np.unique(d["state_index"][d["split"] == "train"]))
    frozen_inputs = {"training_code_sha256": old.digest(__file__),
                     "dataset_sha256": old.digest(OUT / "dataset.npz"),
                     "entity_representation_sha256": old.digest(OUT / "entities.npz"),
                     "protocol_sha256": old.digest(OUT / "protocol.json"),
                     "source_pair_manifest_sha256": old.digest(OUT / "pairs.json"),
                     "normalization_sha256": digest(d["norm"])}
    if state_normalization is not None:
        frozen_inputs["state_normalization_sha256"] = digest(state_normalization)
    x, si, eta = d["x"], d["state_index"], d["eta"]
    cx = d["context"].copy()
    if kind == "additive":
        cx[..., 10:] = 0.
        # Nominal summaries must actually be candidate-independent.
        for c in range(3):
            for state in np.unique(si):
                np.testing.assert_allclose(cx[c, si == state, :10],
                                           np.broadcast_to(cx[c, si == state, :10][0], cx[c, si == state, :10].shape), atol=1e-5)
    model = model_for(kind)
    zero_id = jnp.zeros((1, 3))
    template = model.init(jax.random.PRNGKey(seed), gather(x, si[:1]), jnp.zeros((1, 3)), jnp.zeros((1, 24)), zero_id)
    params = template
    opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(8e-4, weight_decay=1e-4))
    os = opt.init(params)

    @jax.jit
    def step(params, os, xx, ee, cc, ss, ff):
        def loss(pp):
            z = model.apply(pp, xx, ee, cc, jnp.zeros((len(ee), 3))).reshape(3, -1)
            return controller_balanced_nll(z, ss, ff)
        value, grads = jax.value_and_grad(loss)(params)
        updates, os = opt.update(grads, os, params)
        return optax.apply_updates(params, updates), os, value

    predict = jax.jit(lambda p, xx, e, c: model.apply(p, xx, e, c, jnp.zeros((len(e), 3))))
    tr = np.flatnonzero(d["split"] == "train")
    va = np.flatnonzero(d["split"] == "validation")

    def scores(p, indices, variant="correct"):
        z = np.zeros((3, len(eta)), np.float32)
        ci = np.arange(3)
        if variant == "wrong_controller":
            ci = np.roll(ci, 1)
        state_map = np.arange(len(x["agents"]))
        if variant in ("state_shuffle", "state_response_shuffle"):
            active = np.unique(si[indices])
            state_map[active] = np.roll(active, 1)
        imap = np.arange(len(eta))
        if variant in ("context_shuffle", "state_response_shuffle"):
            active = np.unique(si[indices])
            for a, b in zip(active, np.roll(active, 1)):
                imap[np.flatnonzero(si == a)] = np.flatnonzero(si == b)
        emap = np.arange(len(eta))
        if variant == "eta_shuffle":
            for state in np.unique(si[indices]):
                ii = np.flatnonzero(si == state)
                emap[ii] = np.roll(ii, 1)
        for c in range(3):
            for low in range(0, len(indices), 128):
                ii = indices[low:low+128]
                z[c, ii] = np.asarray(predict(p, gather(x, state_map[si[ii]]), jnp.asarray(eta[emap[ii]]), jnp.asarray(cx[ci[c], imap[ii]])))
        return z

    best = (float("inf"), None, 0)
    history = []
    stale = 0
    rng = np.random.default_rng(seed)
    # Exact same draw/order, optimizer, budget and eval cadence across variants.
    for iteration in range(1, 1501):
        draw = rng.choice(tr, 32)
        ii = np.tile(draw, 3)
        cc = np.repeat(np.arange(3), 32)
        params, os, loss = step(params, os, gather(x, si[ii]), jnp.asarray(eta[ii]),
                               jnp.asarray(cx[cc, ii]), jnp.asarray(d["success"][:, draw]), jnp.asarray(d["failure"][:, draw]))
        if iteration % 100:
            continue
        z = scores(params, va)
        result = metrics(z, d, va)
        value = result["NLL"]
        history.append({"step": iteration, "training_objective": float(loss), **result})
        if value < best[0]-1e-5:
            best = (value, serialization.to_bytes(params), iteration)
            stale = 0
        else:
            stale += 1
        # Train equal fixed steps for causal comparison; best checkpoint still
        # uses VAL NLL only. Original early-stopping stop point is logged.
        history[-1]["old_early_stop_would_fire"] = bool(stale >= 7 and iteration >= 700)
    params = serialization.from_bytes(template, best[1])
    result = {"kind": kind, "seed": seed, "best_step": best[2], "trained_steps": 1500,
              "frozen_inputs": frozen_inputs,
              "checkpoint_criterion": "source VAL controller-balanced observed-count NLL only",
              "target_labels": 0, "normalization_training_only": True,
              "architecture": "unchanged H20 Critic" if kind != "additive" else "nominal-only A(h,C_nominal)+B(eta) logits"}
    # Check inference on CPU and accelerator with identical weights and float32
    # inputs, not two differently initialized models or normalization paths.
    check_indices = va[:16]
    xx = gather(x, si[check_indices])
    ee, cc = jnp.asarray(eta[check_indices]), jnp.asarray(cx[0, check_indices])
    reference = np.asarray(predict(params, xx, ee, cc))
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        cpu_params = jax.device_put(params, cpu)
        cpu_x = jax.tree_util.tree_map(lambda v: jax.device_put(np.asarray(v), cpu), xx)
        cpu_eta = jax.device_put(np.asarray(ee), cpu)
        cpu_context = jax.device_put(np.asarray(cc), cpu)
        replay = np.asarray(model.apply(cpu_params, cpu_x, cpu_eta, cpu_context, jnp.zeros((len(check_indices), 3))))
    err = float(np.max(np.abs(reference-replay)))
    assert err < 1e-4, f"Backend representation mismatch: {err}"
    result["backend_consistency"] = {"max_logit_error_cpu_vs_training_backend": err, "dtype": "float32", "passed": True}
    values = {}
    for variant in ("correct", "wrong_controller", "state_shuffle", "context_shuffle", "state_response_shuffle", "eta_shuffle"):
        z = scores(params, va, variant)
        values[variant] = z
        result[variant] = metrics(z, d, va)
        if variant != "correct":
            result[variant]["mean_abs_probability_change"] = float(np.abs(np.asarray(jax.nn.sigmoid(z[:, va]))-np.asarray(jax.nn.sigmoid(values["correct"][:, va]))).mean())
    result["training"] = metrics(scores(params, tr), d, tr)
    # Gradients and parameter movement are diagnostics, not gate substitutes.
    ii = np.tile(tr[:32], 3)
    cc = np.repeat(np.arange(3), 32)
    def audit_loss(pp):
        z = model.apply(pp, gather(x, si[ii]), jnp.asarray(eta[ii]), jnp.asarray(cx[cc, ii]), jnp.zeros((len(ii), 3))).reshape(3, -1)
        return controller_balanced_nll(z, jnp.asarray(d["success"][:, tr[:32]]), jnp.asarray(d["failure"][:, tr[:32]]))
    grad = jax.grad(audit_loss)(params)["params"]
    result["parameter_audit"] = {}
    for key in ("physical_encoder", "context_encoder", "eta_encoder"):
        a = jax.tree_util.tree_leaves(template["params"][key])
        b = jax.tree_util.tree_leaves(params["params"][key])
        g = jax.tree_util.tree_leaves(grad[key])
        result["parameter_audit"][key] = {
            "relative_parameter_update": float(np.sqrt(sum(np.sum(np.asarray(u-v)**2) for u, v in zip(a, b)))/max(1e-12, np.sqrt(sum(np.sum(np.asarray(u)**2) for u in a)))),
            "gradient_norm": float(np.sqrt(sum(np.sum(np.asarray(v)**2) for v in g)))}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "checkpoint.msgpack").write_bytes(best[1])
    write(dest / "normalization.json", d["norm"])
    if state_normalization is not None:
        write(dest / "state_normalization.json", state_normalization)
    write(dest / "history.json", history)
    write(dest / "summary.json", result)
    np.savez_compressed(dest / "validation_predictions.npz", **values)
    print({"kind": kind, "seed": seed, "best_step": best[2], "NLL": result["correct"]["NLL"],
           "B15": result["correct"]["selected_B15"], "controller_reversals": result["correct"]["controller_reversals"]}, flush=True)


def gate():
    results = {(k, s): read(OUT / "models" / k / f"seed{s}/summary.json") for k in KINDS for s in SEEDS}
    full = [results["physical_context", s] for s in SEEDS]
    acc = lambda r, key: r[key]["accuracy"] if r[key]["accuracy"] is not None else 0.
    checks = {
        "controller_reversal_evidence": full[0]["correct"]["controller_reversals"]["total"] >= 10 and full[0]["correct"]["controller_reversals"]["families"] >= 3,
        "controller_reversal_accuracy": np.median([acc(r["correct"], "controller_reversals") for r in full]) >= .60,
        "beats_eta_only_two_seeds": sum(r["correct"]["selected_B15"] > results["eta_only", s]["correct"]["selected_B15"] for r, s in zip(full, SEEDS)) >= 2,
        "beats_additive_two_seeds": sum(r["correct"]["selected_B15"] > results["additive", s]["correct"]["selected_B15"] for r, s in zip(full, SEEDS)) >= 2,
        "context_used": np.median([r["wrong_controller"]["NLL"]-r["correct"]["NLL"] for r in full]) >= .02 or np.median([r["correct"]["selected_B15"]-r["wrong_controller"]["selected_B15"] for r in full]) > 0,
        "state_interaction_accuracy": np.median([acc(r["correct"], "state_reversals") for r in full]) >= .60,
        "state_response_shuffle_degrades": np.median([acc(r["correct"], "state_reversals")-acc(r["state_response_shuffle"], "state_reversals") for r in full]) > 0,
    }
    result = {"checks": {k: bool(v) for k, v in checks.items()}, "passed": bool(all(checks.values())),
              "target_labels_opened": False,
              "verdict": "SOURCE_GATE_PASSED" if all(checks.values()) else "SOURCE_GATE_NOT_PASSED",
              "direct_h_shuffle_separately_reported": True,
              "context_can_encode_state_information": True}
    write(OUT / "source_gate.json", result)
    state = read(OUT / "working_state.json")
    state.update(phase="source_validation_complete", source_gate_passed=result["passed"],
                 source_gate_file="source_gate.json", independent_held_controller_test_started=False,
                 required_next_step="independent confirmation and budget review" if result["passed"] else "source-only failure diagnosis; no target opening")
    write(OUT / "working_state.json", state)
    print(result)
    from .summarize_source import main as summarize_source
    summarize_source()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("train", "gate"))
    ap.add_argument("--kind", choices=(*KINDS, STATE_NORMALIZED))
    ap.add_argument("--seed", type=int, choices=SEEDS)
    a = ap.parse_args()
    train(a.kind, a.seed) if a.action == "train" else gate()
