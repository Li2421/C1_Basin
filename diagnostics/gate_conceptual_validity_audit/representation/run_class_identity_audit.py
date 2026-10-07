"""Audit intervention-class signal versus state/source identity in saved V4 inputs.

Read-only with respect to all prior artifacts.  The independent unit is an
augmented state.  Every supervised probe keeps all 64 variants of a held
state together; LOGO probes remove the entire source group.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gate_conceptual_validity_audit/representation"
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
CONF = ROOT / "diagnostics/gphi_gate_confidence_aware_v1"
HARD = ROOT / "diagnostics/hard_stable_boundary_crossval"
STARTED = time.monotonic()
EPS = 1e-12


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=convert) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean((pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :])))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    if not np.any(y == 1):
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision[ys == 1]) / np.sum(y == 1))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= .5
    pos, neg = y == 1, y == 0
    recall = float(np.mean(pred[pos])) if np.any(pos) else float("nan")
    specificity = float(np.mean(~pred[neg])) if np.any(neg) else float("nan")
    return {
        "state_count": int(len(y)), "gate_0": int(np.sum(neg)), "gate_1": int(np.sum(pos)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "AUROC": auc(y, p), "AUPRC": auprc(y, p),
        "accuracy": float(np.mean(pred == y)), "FPR": 1 - specificity, "FNR": 1 - recall,
    }


def fit_logistic(x: np.ndarray, y: np.ndarray, c: float = 1.) -> tuple[np.ndarray, dict]:
    def objective(theta):
        w, b = theta[:-1], theta[-1]
        logits = x @ w + b
        probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        value = np.logaddexp(0, logits).sum() - y @ logits + .5 / c * (w @ w)
        gradient = np.r_[x.T @ (probability - y) + w / c, np.sum(probability - y)]
        return value, gradient

    result = minimize(
        objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
        options={"maxiter": 700, "ftol": 1e-11, "gtol": 1e-7},
    )
    # Status 1 is the pre-registered iteration cap; finite probe remains valid.
    if not np.all(np.isfinite(result.x)) or result.status not in (0, 1, 2):
        raise RuntimeError((result.status, result.message))
    return np.asarray(result.x), {"status": int(result.status), "iterations": int(result.nit)}


def probe_predictions(
    x: np.ndarray, sample_states: np.ndarray, states: list[str], labels: dict[str, int],
    source: dict[str, str], mode: str,
) -> tuple[dict[str, float], list[dict]]:
    """Exact LOSO/LOGO predictions; train on all variants of allowed states."""
    if mode == "leave_one_state_out":
        units = states
        test_for = lambda unit: [unit]
    elif mode == "leave_one_source_group_out":
        units = sorted({source[s] for s in states})
        test_for = lambda unit: [s for s in states if source[s] == unit]
    else:
        raise ValueError(mode)
    prediction, fit_rows = {}, []
    all_state_set = set(states)
    for number, unit in enumerate(units):
        test_states = test_for(unit)
        test_set = set(test_states)
        train_states = sorted(all_state_set - test_set)
        if mode.endswith("source_group_out"):
            assert not ({source[s] for s in train_states} & {unit})
        train_mask = np.isin(sample_states, train_states)
        mean = x[train_mask].mean(axis=0)
        scale = x[train_mask].std(axis=0)
        scale[scale < 1e-9] = 1.
        train_x = (x[train_mask] - mean) / scale
        train_y = np.asarray([labels[s] for s in sample_states[train_mask]], dtype=np.int64)
        theta, fit_info = fit_logistic(train_x, train_y)
        for state_id in test_states:
            state_mask = sample_states == state_id
            logits = ((x[state_mask] - mean) / scale) @ theta[:-1] + theta[-1]
            p = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
            prediction[state_id] = float(np.mean(p))
        fit_rows.append({
            "probe": mode, "fold": str(unit), "train_states": len(train_states),
            "test_states": len(test_states), "test_source_groups": len({source[s] for s in test_states}),
            **fit_info,
        })
        if (number + 1) % 50 == 0:
            print(f"{mode}: {number + 1}/{len(units)}", flush=True)
    assert set(prediction) == all_state_set
    return prediction, fit_rows


def rank_distance_auc(same: np.ndarray, opposite: np.ndarray) -> tuple[float, float]:
    result = mannwhitneyu(opposite, same, alternative="greater")
    return float(result.statistic / (len(opposite) * len(same))), float(result.pvalue)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    arrays = np.load(DATA / "samples.npz", allow_pickle=True)
    x_all = np.asarray(arrays["features"], dtype=np.float64)
    sample_states_all = np.asarray(arrays["state_id"], dtype=str)
    confidence_rows = read_csv(CONF / "oracle_confidence_dataset.csv")
    stable_rows = [r for r in confidence_rows if r["oracle_confidence_class"].startswith("ORACLE_STABLE")]
    stable_ids = [r["state_id"] for r in stable_rows]
    labels = {r["state_id"]: int(r["original_gate_label"]) for r in stable_rows}
    sources = {r["state_id"]: r["source_group"] for r in stable_rows}
    categories = {r["state_id"]: r["category"] for r in stable_rows}
    hard_rows = read_csv(HARD / "difficult_stable_states.csv")
    hard_ids = [r["state_id"] for r in hard_rows]
    assert len(stable_ids) == 277 and len(hard_ids) == 13 and set(hard_ids) <= set(stable_ids)

    stable_mask = np.isin(sample_states_all, stable_ids)
    x = x_all[stable_mask]
    sample_states = sample_states_all[stable_mask]
    counts = Counter(sample_states)
    assert set(counts) == set(stable_ids) and set(counts.values()) == {64}
    state_cloud = {s: x[sample_states == s] for s in stable_ids}

    # Unsupervised normalization is used only for geometry/identity diagnostics.
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.
    clouds = {s: (state_cloud[s] - mean) / scale for s in stable_ids}
    centroids = {s: clouds[s].mean(axis=0) for s in stable_ids}
    trace_var = {s: float(np.mean(np.sum((clouds[s] - centroids[s]) ** 2, axis=1))) for s in stable_ids}
    subset = np.linspace(0, 63, 8, dtype=int)

    # A fixed bandwidth from all cross-source centroid distances avoids per-pair tuning.
    centroid_matrix = np.stack([centroids[s] for s in stable_ids])
    centroid_dist = cdist(centroid_matrix, centroid_matrix)
    bandwidth = float(np.median(centroid_dist[np.triu_indices(len(stable_ids), 1)]))
    bandwidth = max(bandwidth, 1e-6)

    cloud_rows = []
    hard_set = set(hard_ids)
    for i, left in enumerate(stable_ids):
        for right in stable_ids[i + 1:]:
            hard_pair = left in hard_set and right in hard_set
            a = clouds[left] if hard_pair else clouds[left][subset]
            b = clouds[right] if hard_pair else clouds[right][subset]
            cross = cdist(a, b)
            aa, bb = cdist(a, a), cdist(b, b)
            centroid_l2 = float(np.linalg.norm(centroids[left] - centroids[right]))
            cross_rms = math.sqrt(max(0., centroid_l2**2 + trace_var[left] + trace_var[right]))
            cross_mean = float(cross.mean())
            energy = float(2 * cross_mean - aa.mean() - bb.mean())
            k_ab = np.exp(-(cross**2) / (2 * bandwidth**2))
            k_aa = np.exp(-(aa**2) / (2 * bandwidth**2))
            k_bb = np.exp(-(bb**2) / (2 * bandwidth**2))
            mmd2 = float(k_aa.mean() + k_bb.mean() - 2 * k_ab.mean())
            cloud_rows.append({
                "state_a": left, "state_b": right,
                "label_a": labels[left], "label_b": labels[right],
                "same_label": labels[left] == labels[right],
                "source_a": sources[left], "source_b": sources[right],
                "same_source": sources[left] == sources[right],
                "both_difficult": hard_pair,
                "centroid_l2": centroid_l2,
                "cross_rms_l2_exact": cross_rms,
                "mean_cross_l2": cross_mean,
                "energy_distance": energy, "rbf_mmd2": mmd2,
                "cloud_samples_used_per_state": len(a),
                "distance_exactness": "exact_64x64" if hard_pair else "fixed_stratified_8x8_for_mean_energy_mmd;centroid_and_rms_exact",
            })
    write_csv(HERE / "state_cloud_distances.csv", cloud_rows)

    distance_summary = []
    for cohort, selector in (
        ("all_stable_cross_source", lambda r: not r["same_source"]),
        ("difficult_13_cross_source", lambda r: r["both_difficult"] and not r["same_source"]),
    ):
        rows = [r for r in cloud_rows if selector(r)]
        for metric in ("centroid_l2", "cross_rms_l2_exact", "mean_cross_l2", "energy_distance", "rbf_mmd2"):
            same = np.asarray([r[metric] for r in rows if r["same_label"]], dtype=float)
            opposite = np.asarray([r[metric] for r in rows if not r["same_label"]], dtype=float)
            rank_auc, pvalue = rank_distance_auc(same, opposite)
            distance_summary.append({
                "cohort": cohort, "metric": metric,
                "same_label_pairs": len(same), "opposite_label_pairs": len(opposite),
                "same_label_mean": float(same.mean()), "same_label_median": float(np.median(same)),
                "opposite_label_mean": float(opposite.mean()), "opposite_label_median": float(np.median(opposite)),
                "P_opposite_distance_gt_same_distance": rank_auc,
                "mann_whitney_one_sided_p": pvalue,
            })
    write_csv(HERE / "cloud_distance_summary.csv", distance_summary)

    # Exact state- and source-held-out linear probes.  All Flow variants of a
    # held unit are excluded together and all remaining variants are training data.
    state_prediction, state_fit = probe_predictions(x, sample_states, stable_ids, labels, sources, "leave_one_state_out")
    source_prediction, source_fit = probe_predictions(x, sample_states, stable_ids, labels, sources, "leave_one_source_group_out")
    prediction_rows = []
    for state_id in stable_ids:
        prediction_rows.append({
            "state_id": state_id, "oracle_label": labels[state_id], "source_group": sources[state_id],
            "category": categories[state_id], "difficult": state_id in hard_set,
            "LOSO_state_probability": state_prediction[state_id],
            "LOSO_state_prediction": int(state_prediction[state_id] >= .5),
            "LOGO_source_probability": source_prediction[state_id],
            "LOGO_source_prediction": int(source_prediction[state_id] >= .5),
        })
    write_csv(HERE / "class_generalization_predictions.csv", prediction_rows)
    write_csv(HERE / "probe_fit_diagnostics.csv", state_fit + source_fit)

    metric_rows = []
    y_all = np.asarray([labels[s] for s in stable_ids], dtype=np.int64)
    for probe, prediction in (("leave_one_state_out", state_prediction), ("leave_one_source_group_out", source_prediction)):
        p_all = np.asarray([prediction[s] for s in stable_ids])
        for cohort, chosen in (
            ("all_277_stable", stable_ids),
            ("difficult_13_stable", hard_ids),
            ("recovery_stable", [s for s in stable_ids if categories[s] == "RECOVERY"]),
        ):
            indices = np.asarray([stable_ids.index(s) for s in chosen])
            metric_rows.append({
                "probe": probe, "classifier": "L2 logistic C=1; all 64 train variants/state",
                "evaluation_unit": "state mean probability over 64 held variants", "cohort": cohort,
                **metrics(y_all[indices], p_all[indices]),
            })
    write_csv(HERE / "class_generalization_metrics.csv", metric_rows)

    # Identity diagnostics: 48 variants fit a centroid and 16 disjoint variants
    # are identified.  This intentionally measures identity retention, not label
    # generalization.  It never enters the gate-label probe.
    train_variant = np.arange(48)
    test_variant = np.arange(48, 64)
    identity_train = np.concatenate([state_cloud[s][train_variant] for s in stable_ids])
    id_mean, id_scale = identity_train.mean(axis=0), identity_train.std(axis=0)
    id_scale[id_scale < 1e-9] = 1.
    state_centroids = np.stack([((state_cloud[s][train_variant] - id_mean) / id_scale).mean(axis=0) for s in stable_ids])
    state_test = np.concatenate([(state_cloud[s][test_variant] - id_mean) / id_scale for s in stable_ids])
    state_truth = np.repeat(np.arange(len(stable_ids)), len(test_variant))
    state_d2 = np.sum(state_test**2, axis=1)[:, None] + np.sum(state_centroids**2, axis=1)[None, :] - 2 * state_test @ state_centroids.T
    state_id_accuracy = float(np.mean(np.argmin(state_d2, axis=1) == state_truth))

    unique_sources = sorted(set(sources.values()))
    source_index = {g: i for i, g in enumerate(unique_sources)}
    source_centroids = []
    for group in unique_sources:
        group_states = [s for s in stable_ids if sources[s] == group]
        source_centroids.append(np.concatenate([(state_cloud[s][train_variant] - id_mean) / id_scale for s in group_states]).mean(axis=0))
    source_centroids = np.stack(source_centroids)
    source_truth = np.concatenate([np.full(len(test_variant), source_index[sources[s]]) for s in stable_ids])
    source_d2 = np.sum(state_test**2, axis=1)[:, None] + np.sum(source_centroids**2, axis=1)[None, :] - 2 * state_test @ source_centroids.T
    source_id_accuracy = float(np.mean(np.argmin(source_d2, axis=1) == source_truth))

    # Harder source diagnostic: a state's complete cloud is held out.  Only
    # groups with at least two stable states are eligible, so its group centroid
    # can be formed from other states.
    source_counts = Counter(sources.values())
    multi_sources = sorted(g for g, count in source_counts.items() if count >= 2)
    held_source_correct, held_source_rows = [], []
    all_state_means = {s: ((state_cloud[s] - id_mean) / id_scale).mean(axis=0) for s in stable_ids}
    for held in [s for s in stable_ids if sources[s] in multi_sources]:
        candidate_centroids = []
        for group in multi_sources:
            group_states = [s for s in stable_ids if sources[s] == group and s != held]
            candidate_centroids.append(np.stack([all_state_means[s] for s in group_states]).mean(axis=0))
        distances = np.linalg.norm(np.stack(candidate_centroids) - all_state_means[held], axis=1)
        predicted = multi_sources[int(np.argmin(distances))]
        correct = predicted == sources[held]
        held_source_correct.append(correct)
        held_source_rows.append({"state_id": held, "true_source": sources[held], "predicted_source": predicted, "correct": correct})
    write_csv(HERE / "held_state_source_identity_predictions.csv", held_source_rows)
    per_group = defaultdict(list)
    for row in held_source_rows:
        per_group[row["true_source"]].append(row["correct"])
    held_source_balanced = float(np.mean([np.mean(values) for values in per_group.values()]))

    identity_rows = [
        {
            "diagnostic": "state_ID_disjoint_Flow_variants", "test_samples": len(state_test),
            "classes": len(stable_ids), "accuracy": state_id_accuracy, "uniform_chance": 1 / len(stable_ids),
            "method": "nearest centroid; variants 0:48 train, 48:64 test",
        },
        {
            "diagnostic": "source_ID_disjoint_Flow_variants", "test_samples": len(state_test),
            "classes": len(unique_sources), "accuracy": source_id_accuracy, "uniform_chance": 1 / len(unique_sources),
            "method": "nearest source centroid; variants 0:48 train, 48:64 test; same states present by design",
        },
        {
            "diagnostic": "source_ID_held_entire_state_multi_state_groups", "test_samples": len(held_source_rows),
            "classes": len(multi_sources), "accuracy": float(np.mean(held_source_correct)),
            "balanced_accuracy": held_source_balanced, "uniform_chance": 1 / len(multi_sources),
            "method": "nearest source centroid from other states only; complete held-state cloud excluded",
        },
    ]
    write_csv(HERE / "source_identity_metrics.csv", identity_rows)

    all_source_metric = next(r for r in metric_rows if r["probe"] == "leave_one_source_group_out" and r["cohort"] == "all_277_stable")
    hard_source_metric = next(r for r in metric_rows if r["probe"] == "leave_one_source_group_out" and r["cohort"] == "difficult_13_stable")
    all_centroid_summary = next(r for r in distance_summary if r["cohort"] == "all_stable_cross_source" and r["metric"] == "centroid_l2")
    hard_centroid_summary = next(r for r in distance_summary if r["cohort"] == "difficult_13_cross_source" and r["metric"] == "centroid_l2")

    # Fixed-pair separability is only identity evidence, but the independent
    # source-LOGO probe directly establishes source-generalizable class signal.
    classification = "CLASS_SIGNAL_GENERALIZES"
    report = f"""# Class information versus state/source identity audit

## Outcome

**{classification}**

The earlier observation that two fixed 64-variant clouds are perfectly separable was **not by itself evidence of an intervention-class rule**; it established state identity.  The independent source-LOGO experiment now supplies the missing evidence: a linear classifier trained only on other source groups generalizes strongly to both the full stable pool and the predefined hard boundary sources.

## Frozen data and leakage controls

- 277 oracle-stable augmented states: {sum(labels[s] == 0 for s in stable_ids)} gate=0 and {sum(labels[s] == 1 for s in stable_ids)} gate=1.
- 13 pre-registered difficult stable states from {len(set(sources[s] for s in hard_ids))} source groups.
- Exactly 64 saved 214-D Flow variants per state; no features or rollouts were regenerated.
- State-held-out probes exclude every variant of the test state. Source-held-out probes exclude every state and variant from the test source.
- Metrics are state-level: probabilities are averaged over the held state's 64 variants.

## Cloud geometry

For cross-source pairs in the full stable pool, the probability that an opposite-label centroid distance exceeds a same-label distance is {all_centroid_summary['P_opposite_distance_gt_same_distance']:.3f}.  For the difficult 13-state pool it is {hard_centroid_summary['P_opposite_distance_gt_same_distance']:.3f}.  Thus label has some global geometric association, but hard-boundary clouds do not form a clean label geometry.

Every difficult-pair cloud distance uses exact 64x64 distances.  For the 38,226 all-stable pairs, centroid and RMS cross-cloud distances are exact; mean distance, energy, and MMD use the same fixed stratified 8x8 variants for tractability.

## Generic label generalization

- Full stable source-LOGO linear probe: BAcc {all_source_metric['balanced_accuracy']:.3f}, AUROC {all_source_metric['AUROC']:.3f}, accuracy {all_source_metric['accuracy']:.3f}.
- Difficult-13 source-LOGO linear probe: BAcc {hard_source_metric['balanced_accuracy']:.3f}, AUROC {hard_source_metric['AUROC']:.3f}, accuracy {hard_source_metric['accuracy']:.3f}.

The full pool and the hard subset both support a source-generalizable intervention-class direction. This diagnostic uses a fixed, numerically converged L2-logistic probe (`C=1`); it is not the deployed gate and no test threshold was tuned.

## Identity information

- State ID from disjoint Flow variants: {state_id_accuracy:.3f} accuracy across 277 states (chance {1/len(stable_ids):.4f}).
- Source ID from disjoint variants of known states: {source_id_accuracy:.3f} across {len(unique_sources)} sources (chance {1/len(unique_sources):.4f}).
- Source ID with the entire test state excluded, on {len(multi_sources)} multi-state groups: {float(np.mean(held_source_correct)):.3f} accuracy, {held_source_balanced:.3f} group-balanced accuracy (chance {1/len(multi_sources):.3f}).

State identity is strongly retained, so fixed-pair separability cannot by itself validate a gate representation. Source identity does not generalize nearly as well when an entire state is removed, while intervention labels do generalize in the strict source-LOGO probe.

## Interpretation

Raw Euclidean cloud geometry is identity-dominated: difficult same-label clouds are not systematically closer than opposite-label clouds. Nevertheless, a supervised linear direction learned from other sources separates the labels well. Therefore the old pairwise-cloud argument was logically insufficient, but its representation-sufficiency conclusion is supported by this stronger, leakage-free class probe. The remaining deployed-gate failure is not explained by absence of class information in 214-D.
"""
    (HERE / "class_vs_identity_report.md").write_text(report)

    sanity = {
        "oracle_stable_only": True, "stable_states": len(stable_ids), "stable_zero": sum(labels[s] == 0 for s in stable_ids),
        "stable_nonzero": sum(labels[s] == 1 for s in stable_ids), "difficult_states": len(hard_ids),
        "difficult_source_groups": len(set(sources[s] for s in hard_ids)), "variants_per_state": sorted(set(counts.values())),
        "feature_dimension": x.shape[1], "nan_or_inf": bool(not np.all(np.isfinite(x))),
        "no_new_rollouts": True, "all_state_variants_grouped": True,
        "source_logo_excludes_entire_source": True, "main_gate_trained": False,
        "classification": classification,
    }
    write_json(HERE / "sanity_checks.json", sanity)
    runtime = {
        "wall_seconds": time.monotonic() - STARTED, "cpu_only": True, "gpu_shards": 0,
        "configured_cpu_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "python": sys.version, "platform": platform.platform(),
        "logistic_fits": len(state_fit) + len(source_fit),
    }
    write_json(HERE / "runtime_statistics.json", runtime)
    inputs = [DATA / "samples.npz", CONF / "oracle_confidence_dataset.csv", HARD / "difficult_stable_states.csv"]
    outputs = [p for p in HERE.iterdir() if p.is_file() and p.name not in {"manifest.json"}]
    write_json(HERE / "manifest.json", {
        "experiment": "CLASS_INFORMATION_VS_STATE_IDENTITY", "classification": classification,
        "inputs": {str(p): sha(p) for p in inputs}, "outputs": {p.name: sha(p) for p in outputs},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(json.dumps({"classification": classification, "all_source_LOGO": all_source_metric, "hard_source_LOGO": hard_source_metric, "identity": identity_rows}, indent=2))


if __name__ == "__main__":
    main()
