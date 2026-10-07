"""No new rollout: distinguish state support from neural fitting, source only."""
from itertools import combinations
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from .data import OUT, read, write
from .train import load, model_for, gather, metrics, SEEDS


def main():
    d = load()
    train, val = [np.flatnonzero(d["split"] == split) for split in ("train", "validation")]
    n = d["success"] + d["failure"]
    q = d["success"] / np.maximum(n, 1)
    si = d["state_index"]
    eids = np.array([r["eta_uid"] for r in d["rows"]])
    # Existing unified input only; no label-driven feature engineering.
    h = np.concatenate([v.reshape(len(v), -1) for _, v in sorted(d["x"].items())], axis=1)
    ht = h[np.unique(si[train])]
    scale = ht.std(0)
    varying = scale > 1e-6
    h = (h[:, varying]-ht[:, varying].mean(0)) / np.maximum(scale[varying], .05)
    h /= np.sqrt(h.shape[1])
    cx = d["context"] / np.sqrt(24.)
    predictions, summaries = {}, {}

    for name in ("global_eta_train_mean", "known_controller_eta_train_mean",
                 *[f"{mode}_k{k}" for mode in ("h_known_controller", "C_no_controller_id", "hC_no_controller_id") for k in (1, 5, 11)]):
        z = np.zeros_like(q)
        for i in val:
            same_eta = train[eids[train] == eids[i]]
            assert len(same_eta) == 24
            for c in range(3):
                if name == "global_eta_train_mean":
                    # Equal controller mass; within-controller observed likelihood.
                    prob = np.mean([(d["success"][a, same_eta].sum()+.5)/(n[a, same_eta].sum()+1.) for a in range(3)])
                elif name == "known_controller_eta_train_mean":
                    prob = (d["success"][c, same_eta].sum()+.5)/(n[c, same_eta].sum()+1.)
                else:
                    mode, k = name.rsplit("_k", 1)
                    k = int(k)
                    controller_ids = [c] if mode == "h_known_controller" else range(3)
                    cc = np.repeat(list(controller_ids), len(same_eta))
                    jj = np.tile(same_eta, len(controller_ids))
                    dh = ((h[si[jj]] - h[si[i]])**2).sum(1)
                    dc = ((cx[cc, jj] - cx[c, i])**2).sum(1)
                    distance = dh if mode == "h_known_controller" else dc if mode == "C_no_controller_id" else dh+dc
                    order = np.argsort(distance, kind="stable")[:k]
                    ss = d["success"][cc[order], jj[order]].sum()
                    nn = n[cc[order], jj[order]].sum()
                    prob = (ss+.5)/(nn+1.)
                z[c, i] = np.log(prob)-np.log1p(-prob)
        predictions[name] = z
        m = metrics(z, d, val)
        summaries[name] = {key: m[key] for key in ("NLL", "MAE", "eligible", "selected_B15", "selected_unknown",
                          "controller_reversals", "state_reversals", "selected_observed_Q", "observed_Q_regret")}

    def empirical_reversals(z, ix):
        groups = [[i for i in ix if si[i] == state] for state in np.unique(si[ix])]
        controller_total = controller_right = state_total = state_right = 0
        def sign(c, i, j):
            if min(n[c, i], n[c, j]) < 4:
                return 0
            difference = q[c, i] - q[c, j]
            return 1 if difference >= .5 else -1 if difference <= -.5 else 0
        for ii in groups:
            for a, b in combinations(range(3), 2):
                for i, j in combinations(ii, 2):
                    sa, sb = sign(a, i, j), sign(b, i, j)
                    if sa*sb < 0:
                        controller_total += 1
                        controller_right += int(np.sign(z[a, i]-z[a, j]) == sa and np.sign(z[b, i]-z[b, j]) == sb)
        for aa, bb in combinations(groups, 2):
            for c in range(3):
                for i, j in combinations(range(16), 2):
                    sa, sb = sign(c, aa[i], aa[j]), sign(c, bb[i], bb[j])
                    if sa*sb < 0:
                        state_total += 1
                        state_right += int(np.sign(z[c, aa[i]]-z[c, aa[j]]) == sa and np.sign(z[c, bb[i]]-z[c, bb[j]]) == sb)
        return {"controller_total": controller_total, "controller_correct": controller_right,
                "state_total": state_total, "state_correct": state_right,
                "empirical_margin": .5, "min_observed_trials": 4,
                "caveat": "Training counts mostly Q4; exploratory ordering audit, NOT Q16/B15 certification"}

    fits = []
    for seed in SEEDS:
        dest = OUT / "models/physical_context" / f"seed{seed}"
        model = model_for("physical_context")
        template = model.init(jax.random.PRNGKey(seed), gather(d["x"], si[:1]),
                              jnp.zeros((1, 3)), jnp.zeros((1, 24)), jnp.zeros((1, 3)))
        params = serialization.from_bytes(template, (dest / "checkpoint.msgpack").read_bytes())
        predict = jax.jit(lambda xx, ee, cc: model.apply(params, xx, ee, cc, jnp.zeros((len(ee), 3))))
        z = np.stack([np.asarray(predict(gather(d["x"], si), jnp.asarray(d["eta"]), jnp.asarray(d["context"][c]))) for c in range(3)])
        original = np.load(dest / "validation_predictions.npz")["correct"]
        assert np.max(np.abs(z[:, val]-original[:, val])) < 1e-4
        fits.append({"seed": seed, "training": empirical_reversals(z, train),
                     "validation_same_empirical_rule": empirical_reversals(z, val)})

    result = {"target_labels_read": False, "new_continuations": 0,
              "fixed_diagnostic_k": [1, 5, 11], "selection_or_retraining_from_this_audit": False,
              "scope": "seen source controllers, unseen source families, same fixed eta; not unseen eta/controller/scene",
              "known_controller_baselines": "diagnostic ID-conditioned prior/local-information controls, not a deployable zero-shot model",
              "local_baselines": summaries, "neural_training_fit": fits}
    write(OUT / "source_failure_diagnosis.json", result)
    np.savez_compressed(OUT / "source_local_predictions.npz", **predictions)
    print(result)


if __name__ == "__main__":
    main()
