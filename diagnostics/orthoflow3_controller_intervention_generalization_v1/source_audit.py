"""Source-family validation of context/state/eta use before LOSO target scoring."""
import csv
import json
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.special import expit

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .train import FOLDS, KINDS, OUT, Critic, fold_data, read, write
ALL_KINDS = KINDS + ("contrast_full",)


def csvout(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def run():
    frozen = read(OUT / "models_frozen.json")
    summary = []
    contrasts = []
    for fold in FOLDS:
        npz, rows, ids, state, eta, context, groups, _ = fold_data(fold)
        x = {k: npz[k] for k in ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")}
        s = npz["s"][ids];f = npz["f"][ids]
        for kind in ALL_KINDS:
            seed = frozen["folds"][fold][kind]["selected_seed"]
            folder = OUT / "models" / fold / kind / f"seed{seed}"
            model = Critic(additive=kind == "additive")
            template = model.init(jax.random.PRNGKey(0), gather(x, [0]), jnp.zeros((1, 3)), jnp.zeros((1, 10)))
            params = serialization.from_bytes(template, (folder / "checkpoint.msgpack").read_bytes())
            predict = jax.jit(lambda xx, ee, cc: model.apply(params, xx, ee, cc))
            def infer(ss, ee, cc):
                return np.concatenate([np.asarray(predict(gather(x, ss[j:j + 256]),
                                                       jnp.asarray(ee[j:j + 256]),
                                                       jnp.asarray(cc[j:j + 256])))
                                       for j in range(0, len(ss), 256)])
            for scene, group in groups.items():
                ix = np.concatenate((group["old"]["validation"], group["intervention"]["validation"]))
                perm = np.random.default_rng(2026100440).permutation(len(ix))
                base = infer(state[ix], eta[ix], context[ix])
                tests = {
                    "correct": base,
                    "context_shuffle": infer(state[ix], eta[ix], context[ix][perm]),
                    "state_shuffle": infer(state[ix][perm], eta[ix], context[ix]),
                    "eta_shuffle": infer(state[ix], eta[ix][perm], context[ix]),
                }
                for test, logits in tests.items():
                    likelihood = s[ix] * np.logaddexp(0, -logits) + f[ix] * np.logaddexp(0, logits)
                    summary.append({"fold": fold, "kind": kind, "scene": scene, "test": test,
                                    "pairs": len(ix), "NLL_per_trial": float(likelihood.sum() / max(1, (s[ix] + f[ix]).sum())),
                                    "score_median_abs_change_vs_correct": float(np.median(np.abs(expit(logits) - expit(base))))})
                val_local = set(ix)
                by = defaultdict(dict)
                for j in val_local:
                    row = rows[ids[j]]
                    by[(row["state_uid"], row["eta_uid"])][row["controller_uid"]] = j
                for (uid, eta_uid), controllers in by.items():
                    if len(controllers) < 2:
                        continue
                    jj = list(controllers.values())
                    alt = next((j for j in jj if rows[ids[j]]["role"] == "future_flow_intervention"), None)
                    if alt is None:
                        continue
                    canonical = next((j for j in jj if j != alt), None)
                    if canonical is None:
                        continue
                    z = infer(state[[canonical, alt]], eta[[canonical, alt]], context[[canonical, alt]])
                    base_b15 = (s[canonical] + f[canonical] >= 16 and f[canonical] <= 1)
                    base_non = f[canonical] >= 2
                    alt_b15 = f[alt] <= 1 and s[alt] + f[alt] >= 16
                    alt_non = f[alt] >= 2
                    informative = (base_b15 and alt_non) or (base_non and alt_b15)
                    expected = 1 if base_non and alt_b15 else -1 if base_b15 and alt_non else 0
                    contrasts.append({"fold": fold, "kind": kind, "scene": scene,
                                      "state_uid": uid, "eta_uid": eta_uid,
                                      "base_s": int(s[canonical]), "base_f": int(f[canonical]),
                                      "alternate_s": int(s[alt]), "alternate_f": int(f[alt]),
                                      "base_prediction": float(expit(z[0])), "alternate_prediction": float(expit(z[1])),
                                      "predicted_delta": float(expit(z[1]) - expit(z[0])),
                                      "informative_B15_flip": informative,
                                      "flip_direction_correct": bool(np.sign(z[1] - z[0]) == expected) if informative else None})
    csvout("source_input_use.csv", summary)
    csvout("source_controller_contrasts.csv", contrasts)
    report = {}
    for kind in ALL_KINDS:
        selected = [r for r in contrasts if r["kind"] == kind and r["informative_B15_flip"]]
        report[kind] = {"validation_controller_flips": len(selected),
                        "direction_accuracy": float(np.mean([r["flip_direction_correct"] for r in selected])) if selected else None,
                        "median_abs_predicted_change": float(np.median([abs(r["predicted_delta"]) for r in selected])) if selected else None}
    write(OUT / "source_input_use_summary.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    run()
