"""Strict LOSO gate on the original frozen K16 target candidate pools."""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.special import expit
from scipy.stats import binomtest

from diagnostics.orthoflow3_controller_context_loso_v1.evaluate import TARGET
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .train import FOLDS, KINDS, OUT, OLD, SEEDS, Critic, read, write, sha
ALL_KINDS = KINDS + ("contrast_full",)


def save_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def predict():
    assert (OUT / "models_frozen.json").exists()
    frozen = read(OUT / "models_frozen.json")
    if (OUT / "target_predictions.json").exists():
        return
    result = {}
    for fold in FOLDS:
        manifest = read(TARGET / fold / "manifest.json")
        target_context = np.load(OUT / f"target_nominal_{fold}.npz")
        x = dict(np.load(TARGET / fold / "entities.npz"))
        n = len(manifest)
        eta = np.asarray([m["eta"] for m in manifest], np.float32).reshape(-1, 3)
        state_index = np.repeat(np.arange(n), 16)
        base_c = target_context["context"]
        wrong_c = target_context["wrong_context"]
        rng = np.random.default_rng(2026100427)
        perm = rng.permutation(n)
        scores = {}
        logs = {}
        for kind in ALL_KINDS:
            chosen_seed = frozen["folds"][fold][kind]["selected_seed"]
            for seed in SEEDS:
                folder = OUT / "models" / fold / kind / f"seed{seed}"
                training = read(folder / "training.json")
                assert not training["target_labels_used"]
                assert sha(folder / "checkpoint.msgpack") == training["checkpoint_sha256"]
                norm = read(folder / "normalization.json")
                assert not norm["target_data_used"] and norm["target_scene"] == FOLDS[fold]
                e = (eta - np.asarray(norm["eta_center"], np.float32)) / np.asarray(norm["eta_radius"], np.float32)
                def nc(c):
                    return (c - np.asarray(norm["context_center"], np.float32)) / np.asarray(norm["context_scale"], np.float32)
                c = nc(base_c)
                wrong = nc(wrong_c)
                model = Critic(additive=kind == "additive")
                template = model.init(jax.random.PRNGKey(0), gather(x, [0]), jnp.zeros((1, 3)), jnp.zeros((1, 10)))
                params = serialization.from_bytes(template, (folder / "checkpoint.msgpack").read_bytes())
                fn = jax.jit(lambda xx, ee, cc: model.apply(params, xx, ee, cc))
                def infer(si, ee, cc):
                    z = np.concatenate([np.asarray(fn(gather(x, si[j:j + 128]),
                                                  jnp.asarray(ee[j:j + 128]),
                                                  jnp.asarray(cc[j:j + 128])))
                                        for j in range(0, len(si), 128)])
                    return z.reshape(n, 16)
                z = infer(state_index, e, c[state_index])
                label = f"{kind}_seed{seed}"
                logs[label] = z.tolist()
                scores[label] = expit(z).tolist()
                if seed == chosen_seed:
                    logs[kind] = z.tolist()
                    scores[kind] = expit(z).tolist()
                    tests = {
                        "context_shuffle": (state_index, e, c[perm[state_index]]),
                        "wrong_controller": (state_index, e, wrong[state_index]),
                        "state_shuffle": (perm[state_index], e, c[state_index]),
                        "eta_shuffle": (state_index, np.roll(e.reshape(n, 16, 3), 1, axis=1).reshape(-1, 3), c[state_index]),
                    }
                    for test, (si, ee, cc) in tests.items():
                        zz = infer(si, ee, cc)
                        logs[f"{kind}_{test}"] = zz.tolist()
                        scores[f"{kind}_{test}"] = expit(zz).tolist()
            scores[f"{kind}_ensemble"] = np.mean([scores[f"{kind}_seed{s}"] for s in SEEDS], 0).tolist()
        result[fold] = {"state_uids": [m["state_uid"] for m in manifest],
                        "eta": eta.reshape(n, 16, 3).tolist(),
                        "scores": scores, "logits": logs}
    write(OUT / "target_predictions.json",
          {"models_frozen_sha256": sha(OUT / "models_frozen.json"),
           "target_labels_opened_for_prediction": False,
           "folds": result})


def evaluate():
    predictions = read(OUT / "target_predictions.json")
    truth = read(OLD / "cached_truth.json")
    old = read(OLD / "target_predictions.json")
    results, paired, selected = [], [], []
    for fold in FOLDS:
        pp = predictions["folds"][fold]
        tt = truth[fold]
        assert pp["state_uids"] == [t["state_uid"] for t in tt]
        lo = np.asarray([t["lower"] for t in tt], float)
        hi = np.asarray([t["upper"] for t in tt], float)
        robust = np.asarray([[v is True for v in t["robust"]] for t in tt], bool)
        unknown = np.asarray([[v is None for v in t["robust"]] for t in tt], bool)
        n = len(tt)
        scores = {k: np.asarray(v, float) for k, v in pp["scores"].items()}
        previous = old["folds"][fold]["scores"]
        scores["old_shared"] = np.asarray(previous["partial_count_shared"], float)
        scores["eta_only"] = np.asarray(previous["partial_count_eta_only"], float)
        scores["oracle"] = lo
        chosen = {}
        for method, values in scores.items():
            index = values.argmax(1)
            ii = np.arange(n)
            yes = robust[ii, index]
            unk = unknown[ii, index]
            chosen[method] = (yes, unk)
            predicted = values[ii, index]
            qlo = lo[ii, index]
            qhi = hi[ii, index]
            results.append({"fold": fold, "method": method,
                            "N": n, "B15": int(yes.sum()), "unknown": int(unk.sum()),
                            "oracle_B15": int(robust.any(1).sum()),
                            "oracle_gap_lower": int(robust.any(1).sum() - yes.sum() - unk.sum()),
                            "mean_Q_lower": float(qlo.mean()), "mean_Q_upper": float(qhi.mean()),
                            "selected_predicted_Q": float(predicted.mean()),
                            "severe_FP_p_gt_0_9_true_Q_le_0_5": int(((predicted > .9) & (qhi <= .5)).sum()),
                            "p_gt_0_95_B15_precision": float(yes[predicted > .95].mean()) if np.any(predicted > .95) else None,
                            "top1_tie_states": int(((values == values.max(1)[:, None]).sum(1) > 1).sum())})
            for i, j in enumerate(index):
                selected.append({"fold": fold, "method": method, "state_uid": tt[i]["state_uid"],
                                 "proposal_index": int(j), "predicted_Q": float(predicted[i]),
                                 "Q_lower": float(qlo[i]), "Q_upper": float(qhi[i]),
                                 "B15": bool(yes[i]), "unresolved": bool(unk[i])})
        for main in ("structured_full", "additive", "contrast_full"):
            for control in ("eta_only", "old_shared", f"{main}_context_shuffle",
                            f"{main}_wrong_controller", f"{main}_state_shuffle",
                            f"{main}_eta_shuffle"):
                a, ua = chosen[main]
                b, ub = chosen[control]
                valid = ~(ua | ub)
                rescue = int((a & ~b & valid).sum())
                breaks = int((b & ~a & valid).sum())
                paired.append({"fold": fold, "main": main, "control": control,
                               "rescue": rescue, "break": breaks, "resolved": int(valid.sum()),
                               "p_two_sided": float(binomtest(rescue, rescue + breaks).pvalue) if rescue + breaks else 1.})
    save_csv("loso_results.csv", results)
    save_csv("paired_comparisons.csv", paired)
    save_csv("selected_proposals.csv", selected)
    write(OUT / "evaluation_status.json", {"strict_frozen_test_opened_after_model_freeze": True,
                                            "results": "loso_results.csv", "paired": "paired_comparisons.csv"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("predict", "evaluate"))
    args = parser.parse_args()
    globals()[args.action]()
