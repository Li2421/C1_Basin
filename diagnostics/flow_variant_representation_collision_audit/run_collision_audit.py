"""Offline audit of saved Flow-variant input clouds for fixed state pairs.

This script is intentionally read-only with respect to all prior experiments.  It
uses only exact saved 214-D inputs; it does not generate states, rollouts, labels,
or fit a learned model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOCAL = ROOT / "diagnostics/hard_stable_local_feature_audit"
CV = ROOT / "diagnostics/hard_stable_boundary_crossval"
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
ORACLE = ROOT / "diagnostics/oracle_boundary_confidence_audit"

START = time.perf_counter()
START_UTC = datetime.now(timezone.utc)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    keys = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2))


def within_values(distance: np.ndarray) -> np.ndarray:
    return distance[np.triu_indices(len(distance), k=1)]


def cloud_geometry(h0: np.ndarray, h1: np.ndarray) -> dict:
    d00 = rms_distance_matrix(h0, h0)
    d11 = rms_distance_matrix(h1, h1)
    d01 = rms_distance_matrix(h0, h1)
    w0 = within_values(d00)
    w1 = within_values(d11)
    n0_cross = d01.min(axis=1)
    n1_cross = d01.min(axis=0)
    d00_masked = d00.copy()
    d11_masked = d11.copy()
    np.fill_diagonal(d00_masked, np.inf)
    np.fill_diagonal(d11_masked, np.inf)
    n0_within = d00_masked.min(axis=1)
    n1_within = d11_masked.min(axis=1)
    ratios = np.r_[n0_cross / np.maximum(n0_within, 1e-15), n1_cross / np.maximum(n1_within, 1e-15)]
    opposite = np.r_[n0_cross < n0_within, n1_cross < n1_within]
    typical_within = float((w0.mean() + w1.mean()) / 2)
    return {
        "within_h0_mean": float(w0.mean()),
        "within_h0_median": float(np.median(w0)),
        "within_h1_mean": float(w1.mean()),
        "within_h1_median": float(np.median(w1)),
        "within_pooled_mean": typical_within,
        "within_pooled_median": float(np.median(np.r_[w0, w1])),
        "cross_mean": float(d01.mean()),
        "cross_median": float(np.median(d01)),
        "cross_to_within_mean_ratio": float(d01.mean() / max(typical_within, 1e-15)),
        "nearest_cross_mean": float(np.r_[n0_cross, n1_cross].mean()),
        "nearest_within_mean": float(np.r_[n0_within, n1_within].mean()),
        "nearest_cross_over_within_mean_ratio": float(ratios.mean()),
        "nearest_cross_over_within_median_ratio": float(np.median(ratios)),
        "opposite_state_nearest_neighbor_rate": float(opposite.mean()),
        "d00": d00,
        "d11": d11,
        "d01": d01,
        "nearest_ratios": ratios,
        "opposite_flags": opposite,
    }


def knn_accuracy(h0: np.ndarray, h1: np.ndarray, k: int) -> float:
    x = np.r_[h0, h1]
    y = np.r_[np.zeros(len(h0), int), np.ones(len(h1), int)]
    distance = rms_distance_matrix(x, x)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1)[:, :k]
    prediction = (y[neighbors].mean(axis=1) > 0.5).astype(int)
    return float(np.mean(prediction == y))


def energy_stat(distance: np.ndarray, labels: np.ndarray) -> float:
    a = labels == 0
    b = ~a
    return float(
        2 * distance[np.ix_(a, b)].mean()
        - distance[np.ix_(a, a)].mean()
        - distance[np.ix_(b, b)].mean()
    )


def mmd_stat(kernel: np.ndarray, labels: np.ndarray) -> float:
    a = labels == 0
    b = ~a
    return float(
        kernel[np.ix_(a, a)].mean()
        + kernel[np.ix_(b, b)].mean()
        - 2 * kernel[np.ix_(a, b)].mean()
    )


def two_sample_tests(h0: np.ndarray, h1: np.ndarray, permutations: int, seed: int) -> dict:
    x = np.r_[h0, h1]
    labels = np.r_[np.zeros(len(h0), int), np.ones(len(h1), int)]
    distance = rms_distance_matrix(x, x)
    offdiag = distance[~np.eye(len(x), dtype=bool)]
    bandwidth = float(np.median(offdiag[offdiag > 0]))
    bandwidth = max(bandwidth, 1e-12)
    kernel = np.exp(-(distance**2) / (2 * bandwidth**2))
    observed_energy = energy_stat(distance, labels)
    observed_mmd = mmd_stat(kernel, labels)
    rng = np.random.default_rng(seed)
    ge_energy = 0
    ge_mmd = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        ge_energy += energy_stat(distance, shuffled) >= observed_energy - 1e-15
        ge_mmd += mmd_stat(kernel, shuffled) >= observed_mmd - 1e-15
    return {
        "energy_distance": observed_energy,
        "energy_permutation_p": (ge_energy + 1) / (permutations + 1),
        "mmd2_biased": observed_mmd,
        "mmd_permutation_p": (ge_mmd + 1) / (permutations + 1),
        "mmd_rbf_bandwidth_normalized_rms": bandwidth,
        "permutations": permutations,
    }


def pca_coordinates(x: np.ndarray, components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :components] * s[:components]
    explained = s**2 / max(float(np.sum(s**2)), 1e-15)
    return coords, explained[:components]


def render_pca(pairs: list[dict], clouds: dict[str, tuple[np.ndarray, np.ndarray]], seeds: dict, path: Path) -> None:
    width, panel_h = 1500, 430
    image = Image.new("RGB", (width, panel_h * len(pairs)), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = [(35, 100, 210), (220, 75, 55)]
    for row_index, pair in enumerate(pairs):
        h0, h1 = clouds[pair["pair_id"]]
        x = np.r_[h0, h1]
        coords, explained = pca_coordinates(x)
        y0 = row_index * panel_h
        draw.text((12, y0 + 8), f"{pair['pair_id']}: blue=gate0, red=gate1", fill="black", font=font)
        for col, axes in enumerate(((0, 1), (0, 2))):
            left = 70 + col * 730
            top = y0 + 50
            plot_w, plot_h = 630, 330
            vals = coords[:, list(axes)]
            low = vals.min(axis=0)
            high = vals.max(axis=0)
            span = np.maximum(high - low, 1e-12)
            px = left + 15 + (vals[:, 0] - low[0]) / span[0] * (plot_w - 30)
            py = top + plot_h - 15 - (vals[:, 1] - low[1]) / span[1] * (plot_h - 30)
            draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(160, 160, 160))
            common = seeds[pair["pair_id"]]
            if common:
                index0 = {int(v): i for i, v in enumerate(pair["seeds0"])}
                index1 = {int(v): i for i, v in enumerate(pair["seeds1"])}
                for seed_value in common:
                    a = index0[int(seed_value)]
                    b = len(h0) + index1[int(seed_value)]
                    draw.line((float(px[a]), float(py[a]), float(px[b]), float(py[b])), fill=(220, 220, 220), width=1)
            for label, offset, count in ((0, 0, len(h0)), (1, len(h0), len(h1))):
                for idx in range(offset, offset + count):
                    r = 3
                    draw.ellipse((float(px[idx] - r), float(py[idx] - r), float(px[idx] + r), float(py[idx] + r)), fill=colors[label])
            draw.text(
                (left + 8, top + 7),
                f"PC{axes[0]+1} vs PC{axes[1]+1}; EV={explained[axes[0]]:.2%}/{explained[axes[1]]:.2%}",
                fill="black",
                font=font,
            )
    image.save(path)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    local_rows = read_csv(LOCAL / "per_state_failure_classification.csv")
    local_by_id = {row["state_id"]: row for row in local_rows}
    collision_zero_ids = [
        "RBV_Q_pair228_m080_s95401003_p030",
        "RB_Q_pair226_m080_s95400802_p073",
    ]
    for state_id in collision_zero_ids:
        assert local_by_id[state_id]["failure_classification"] == "REPRESENTATION_COLLISION"

    # The control is fixed before this analysis from the previous oracle audit:
    # boundary_pair_10 was an already-identified, clearly-separated stable pair
    # involving a non-collision zero state. No metric from this audit selects it.
    oracle_pairs = {row["pair_id"]: row for row in read_csv(ORACLE / "matched_pair_probability_differences.csv")}
    control = oracle_pairs["boundary_pair_10"]
    assert control["separation_class"] == "CLEARLY_SEPARATED"

    pairs = []
    for i, zero_id in enumerate(collision_zero_ids, 1):
        source = local_by_id[zero_id]
        pairs.append(
            {
                "pair_id": f"collision_pair_{i}",
                "pair_role": "primary_collision",
                "zero_state_id": zero_id,
                "nonzero_state_id": source["nearest_opposite_label_state"],
                "zero_source_group": source["source_group"],
                "prior_state_level_opposite_distance": float(source["nearest_opposite_label_distance"]),
                "selection_source": "hard_stable_local_feature_audit/per_state_failure_classification.csv",
                "selection_rule": "fixed nearest oracle-stable opposite-label neighbor from previous audit",
            }
        )
    pairs.append(
        {
            "pair_id": "control_pair_boundary_10",
            "pair_role": "preexisting_noncollision_control",
            "zero_state_id": control["zero_state_id"],
            "nonzero_state_id": control["nonzero_state_id"],
            "zero_source_group": control["source_group_zero"],
            "prior_state_level_opposite_distance": float(control["feature_distance"]),
            "selection_source": "oracle_boundary_confidence_audit/matched_pair_probability_differences.csv",
            "selection_rule": "pre-existing CLEARLY_SEPARATED stable pair boundary_pair_10; selected before cloud metrics",
        }
    )

    arrays = np.load(DATA / "samples.npz", allow_pickle=True)
    features = arrays["features"].astype(np.float64)
    state_ids = arrays["state_id"].astype(str)
    flow_seeds = arrays["flow_seed"].astype(np.int64)
    assert features.shape[1] == 214
    schema = read_json(DATA / "feature_schema.json")
    folds = read_json(CV / "fold_manifest.json")["folds"]
    fold_by_group = {fold["heldout_source_group"]: fold for fold in folds}

    binary = np.zeros(214, bool)
    slices = {}
    for segment in schema["segments"]:
        start = int(segment["offset"])
        stop = start + int(segment["length"])
        slices[segment["name"]] = np.arange(start, stop)
        if segment["unit"] == "boolean":
            binary[start:stop] = True

    groups = {
        "geometry_observation": ["observation", "positions", "last_executed_velocities", "wall_barrier_h"],
        "goal_relative": ["goal_relative", "B_goal", "goal_errors"],
        "inter_agent_relative": ["inter_agent_relative_position", "inter_agent_relative_velocity", "B_rel", "pairwise_barrier_h"],
        "history_monitor": [
            "recent_progress_2s", "window_ready", "candidate_active", "candidate_since_step",
            "candidate_age", "stuck_timer", "max_stuck_timer", "ever_candidate_deadlock",
            "history_start_step", "goal_error_history_tail_41",
        ],
        "episode_time": ["physical_timestep", "episode_time", "normalized_episode_step", "normalized_remaining_horizon"],
        "u_flow": ["u_flow"],
        "u_safe": ["u_safe"],
        "control_projection": [
            "first_projection_delta", "first_projection_delta_norm", "first_projection_linear_residuals",
            "first_projection_active_linear", "u_safe_agent_speeds", "first_projection_active_speed",
        ],
    }
    group_indices = {name: np.concatenate([slices[item] for item in items]) for name, items in groups.items()}
    assert sorted(np.concatenate(list(group_indices.values())).tolist()) == list(range(214))

    inventory_rows = []
    static_rows = []
    cloud_rows = []
    nearest_rows = []
    separability_rows = []
    matched_rows = []
    group_rows = []
    pair_output_rows = []
    clouds = {}
    common_seeds_by_pair = {}
    normalizations = {}
    pair_metrics = {}

    for pair_number, pair in enumerate(pairs):
        fold = fold_by_group[pair["zero_source_group"]]
        train_ids = set(fold["normalization_fit_state_ids"])
        train_mask = np.array([state_id in train_ids for state_id in state_ids])
        mean = features[train_mask].mean(axis=0)
        scale = features[train_mask].std(axis=0)
        scale[scale < 1e-8] = 1.0
        mean[binary] = 0.0
        scale[binary] = 1.0
        normalizations[pair["pair_id"]] = (mean, scale)
        assert not any(state_ids[train_mask] == pair["zero_state_id"])

        state_clouds = []
        state_seed_arrays = []
        state_static_masks = []
        for role, state_id in (("gate0", pair["zero_state_id"]), ("gate1", pair["nonzero_state_id"])):
            mask = state_ids == state_id
            h = features[mask]
            seed_values = flow_seeds[mask]
            order = np.argsort(seed_values)
            h = h[order]
            seed_values = seed_values[order]
            normalized = (h - mean) / scale
            finite = bool(np.isfinite(normalized).all())
            unique_seeds = len(np.unique(seed_values))
            static_mask = np.ptp(h, axis=0) <= 1e-12
            inventory_rows.append(
                {
                    "pair_id": pair["pair_id"], "pair_role": pair["pair_role"], "state_role": role,
                    "state_id": state_id, "variant_count": len(h), "dimension": h.shape[1] if len(h) else None,
                    "unique_seed_count": unique_seeds, "seed_min": int(seed_values.min()), "seed_max": int(seed_values.max()),
                    "finite": finite, "state_static_dimensions": int(static_mask.sum()),
                    "flow_dependent_dimensions": int((~static_mask).sum()),
                }
            )
            assert len(h) == 64 and h.shape[1] == 214 and unique_seeds == len(h) and finite
            state_clouds.append(normalized)
            state_seed_arrays.append(seed_values)
            state_static_masks.append(static_mask)

        h0, h1 = state_clouds
        seeds0, seeds1 = state_seed_arrays
        pair["seeds0"], pair["seeds1"] = seeds0, seeds1
        clouds[pair["pair_id"]] = (h0, h1)
        common_seeds = sorted(set(seeds0.tolist()) & set(seeds1.tolist()))
        common_seeds_by_pair[pair["pair_id"]] = common_seeds
        common_static = state_static_masks[0] & state_static_masks[1]
        flow_union = ~common_static
        static_distance = float(np.sqrt(np.mean((h0[:, common_static].mean(0) - h1[:, common_static].mean(0)) ** 2))) if common_static.any() else None
        flow_centroid_distance = float(np.sqrt(np.mean((h0[:, flow_union].mean(0) - h1[:, flow_union].mean(0)) ** 2))) if flow_union.any() else None
        flow_geom = cloud_geometry(h0[:, flow_union], h1[:, flow_union]) if flow_union.any() else None
        static_rows.append(
            {
                "pair_id": pair["pair_id"], "common_state_static_dimensions": int(common_static.sum()),
                "union_flow_dependent_dimensions": int(flow_union.sum()),
                "static_component_centroid_distance_normalized_rms": static_distance,
                "flow_component_centroid_distance_normalized_rms": flow_centroid_distance,
                "flow_component_within_pooled_mean_distance": flow_geom["within_pooled_mean"] if flow_geom else None,
                "flow_component_cross_mean_distance": flow_geom["cross_mean"] if flow_geom else None,
                "flow_component_cross_to_within_ratio": flow_geom["cross_to_within_mean_ratio"] if flow_geom else None,
            }
        )

        geometry = cloud_geometry(h0, h1)
        tests = two_sample_tests(h0, h1, permutations=1999, seed=8400 + pair_number)
        k_accuracy = {k: knn_accuracy(h0, h1, k) for k in (1, 3, 5)}
        metrics = {k: v for k, v in geometry.items() if not isinstance(v, np.ndarray)} | tests | {
            "knn_1_accuracy": k_accuracy[1], "knn_3_accuracy": k_accuracy[3], "knn_5_accuracy": k_accuracy[5]
        }
        pair_metrics[pair["pair_id"]] = metrics
        cloud_rows.append({"pair_id": pair["pair_id"], **{k: metrics[k] for k in metrics if k.startswith("within") or k.startswith("cross") or k.startswith("nearest")}})
        for state_role, ratios, flags in (
            ("gate0", geometry["nearest_ratios"][:len(h0)], geometry["opposite_flags"][:len(h0)]),
            ("gate1", geometry["nearest_ratios"][len(h0):], geometry["opposite_flags"][len(h0):]),
        ):
            nearest_rows.append(
                {
                    "pair_id": pair["pair_id"], "state_role": state_role, "sample_count": len(ratios),
                    "cross_nearest_over_within_nearest_mean": float(ratios.mean()),
                    "cross_nearest_over_within_nearest_median": float(np.median(ratios)),
                    "opposite_state_nearest_neighbor_rate": float(flags.mean()),
                }
            )
        separability_rows.extend(
            [
                {"pair_id": pair["pair_id"], "diagnostic": f"LOO_{k}NN", "value": k_accuracy[k], "p_value": "", "detail": "state-identity accuracy"}
                for k in (1, 3, 5)
            ]
            + [
                {"pair_id": pair["pair_id"], "diagnostic": "energy_distance", "value": tests["energy_distance"], "p_value": tests["energy_permutation_p"], "detail": "1999 label permutations"},
                {"pair_id": pair["pair_id"], "diagnostic": "MMD_RBF_biased", "value": tests["mmd2_biased"], "p_value": tests["mmd_permutation_p"], "detail": f"median-heuristic bandwidth={tests['mmd_rbf_bandwidth_normalized_rms']:.8g}; 1999 permutations"},
            ]
        )

        if common_seeds:
            index0 = {int(v): i for i, v in enumerate(seeds0)}
            index1 = {int(v): i for i, v in enumerate(seeds1)}
            values = []
            for seed_value in common_seeds:
                distance = float(np.sqrt(np.mean((h0[index0[seed_value]] - h1[index1[seed_value]]) ** 2)))
                values.append(distance)
                matched_rows.append({"pair_id": pair["pair_id"], "flow_seed": seed_value, "matched_cross_state_distance": distance, "record_type": "per_seed"})
            matched_rows.append(
                {
                    "pair_id": pair["pair_id"], "flow_seed": "ALL", "matched_cross_state_distance": float(np.mean(values)),
                    "record_type": "summary", "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values)),
                    "typical_within_state_flow_distance": geometry["within_pooled_mean"],
                    "matched_cross_to_within_ratio": float(np.mean(values) / max(geometry["within_pooled_mean"], 1e-15)),
                }
            )
        else:
            matched_rows.append(
                {
                    "pair_id": pair["pair_id"], "flow_seed": "NONE", "matched_cross_state_distance": "",
                    "record_type": "unavailable", "note": "No exact shared Flow seed IDs; no index-based pseudo-matching performed.",
                    "typical_within_state_flow_distance": geometry["within_pooled_mean"],
                }
            )

        clean_groups = []
        for group_number, (group_name, indices) in enumerate(group_indices.items()):
            gh0, gh1 = h0[:, indices], h1[:, indices]
            group_geom = cloud_geometry(gh0, gh1)
            group_tests = two_sample_tests(gh0, gh1, permutations=499, seed=9400 + pair_number * 100 + group_number)
            group_knn = {k: knn_accuracy(gh0, gh1, k) for k in (1, 3, 5)}
            if group_knn[1] >= 0.9 and group_tests["mmd_permutation_p"] <= 0.05:
                clean_groups.append(group_name)
            group_rows.append(
                {
                    "pair_id": pair["pair_id"], "feature_group": group_name, "dimension_count": len(indices),
                    "within_pooled_mean_distance": group_geom["within_pooled_mean"],
                    "between_centroid_distance": float(np.sqrt(np.mean((gh0.mean(0) - gh1.mean(0)) ** 2))),
                    "cross_mean_distance": group_geom["cross_mean"],
                    "cross_to_within_ratio": group_geom["cross_to_within_mean_ratio"],
                    "opposite_state_nearest_neighbor_rate": group_geom["opposite_state_nearest_neighbor_rate"],
                    "knn_1_accuracy": group_knn[1], "knn_3_accuracy": group_knn[3], "knn_5_accuracy": group_knn[5],
                    "energy_distance": group_tests["energy_distance"], "energy_p": group_tests["energy_permutation_p"],
                    "mmd2_biased": group_tests["mmd2_biased"], "mmd_p": group_tests["mmd_permutation_p"],
                }
            )

        if pair["pair_role"] == "preexisting_noncollision_control":
            classification = (
                "CONTROL_CLEANLY_SEPARATED"
                if min(k_accuracy.values()) >= 0.90
                and geometry["opposite_state_nearest_neighbor_rate"] <= 0.10
                and tests["mmd_permutation_p"] <= 0.01
                else "CONTROL_NOT_CLEANLY_SEPARATED"
            )
        elif (
            k_accuracy[1] <= 0.65 and k_accuracy[3] <= 0.65 and k_accuracy[5] <= 0.65
            and geometry["opposite_state_nearest_neighbor_rate"] >= 0.35
            and geometry["cross_to_within_mean_ratio"] <= 1.25 and not clean_groups
        ):
            classification = "TRUE_DISTRIBUTIONAL_COLLISION"
        elif (
            min(k_accuracy.values()) >= 0.90
            and geometry["opposite_state_nearest_neighbor_rate"] <= 0.10
            and tests["mmd_permutation_p"] <= 0.01 and clean_groups
        ):
            classification = "STATE_SUMMARY_COLLISION_ONLY"
        else:
            classification = "PARTIAL_DISTRIBUTIONAL_OVERLAP"
        pair["classification"] = classification
        pair["cleanly_separating_feature_groups"] = "|".join(clean_groups)
        pair["variant_count_gate0"] = len(h0)
        pair["variant_count_gate1"] = len(h1)
        pair["matched_seed_count"] = len(common_seeds)
        pair["normalization_fold"] = fold["fold_id"]
        pair_output_rows.append({k: v for k, v in pair.items() if k not in {"seeds0", "seeds1"}})

    collision_classes = [pair["classification"] for pair in pairs if pair["pair_role"] == "primary_collision"]
    if all(value == "TRUE_DISTRIBUTIONAL_COLLISION" for value in collision_classes):
        overall = "TRUE_REPRESENTATION_COLLISION_SUPPORTED"
    elif all(value == "STATE_SUMMARY_COLLISION_ONLY" for value in collision_classes):
        overall = "REPRESENTATION_INFORMATION_PRESENT"
    else:
        overall = "MIXED_FLOW_VARIANT_COLLISION_RESULT"

    render_pca(pairs, clouds, common_seeds_by_pair, HERE / "pca_projection.png")
    write_csv(HERE / "audited_pairs.csv", pair_output_rows)
    write_csv(HERE / "flow_variant_inventory.csv", inventory_rows)
    write_csv(HERE / "static_vs_flow_features.csv", static_rows)
    write_csv(HERE / "cloud_distance_statistics.csv", cloud_rows)
    write_csv(HERE / "nearest_neighbor_overlap.csv", nearest_rows)
    write_csv(HERE / "nonparametric_separability.csv", separability_rows)
    write_csv(HERE / "matched_seed_distances.csv", matched_rows)
    write_csv(HERE / "feature_group_overlap.csv", group_rows)
    control_rows = []
    for pair in pairs:
        metric = pair_metrics[pair["pair_id"]]
        control_rows.append(
            {
                "pair_id": pair["pair_id"], "pair_role": pair["pair_role"], "classification": pair["classification"],
                "within_pooled_mean": metric["within_pooled_mean"], "cross_mean": metric["cross_mean"],
                "cross_to_within_ratio": metric["cross_to_within_mean_ratio"],
                "opposite_nearest_neighbor_rate": metric["opposite_state_nearest_neighbor_rate"],
                "knn_1_accuracy": metric["knn_1_accuracy"], "knn_3_accuracy": metric["knn_3_accuracy"], "knn_5_accuracy": metric["knn_5_accuracy"],
                "energy_distance": metric["energy_distance"], "energy_p": metric["energy_permutation_p"],
                "mmd2": metric["mmd2_biased"], "mmd_p": metric["mmd_permutation_p"],
            }
        )
    write_csv(HERE / "control_pair_comparison.csv", control_rows)

    sanity = {
        "passed": True,
        "feature_dimension_exactly_214": features.shape[1] == 214,
        "all_states_have_64_variants": all(row["variant_count"] == 64 for row in inventory_rows),
        "all_seed_ids_unique_within_state": all(row["unique_seed_count"] == row["variant_count"] for row in inventory_rows),
        "all_features_finite": all(row["finite"] for row in inventory_rows),
        "fixed_collision_neighbors_match_previous_audit": all(
            pair["nonzero_state_id"] == local_by_id[pair["zero_state_id"]]["nearest_opposite_label_state"]
            for pair in pairs[:2]
        ),
        "control_selected_without_current_results": True,
        "control_selection_record": "pre-existing boundary_pair_10 / CLEARLY_SEPARATED",
        "normalization_train_only": True,
        "binary_features_left_literal": True,
        "no_new_states": 0,
        "no_new_oracle_rollouts": 0,
        "models_trained": 0,
        "collision_pairs_exact_matched_seed_counts": {pair["pair_id"]: pair["matched_seed_count"] for pair in pairs[:2]},
        "warning": "No exact matched Flow seeds exist for the two collision pairs; matched-seed claims are therefore not made for them.",
    }
    write_json(HERE / "sanity_checks.json", sanity)

    def fmt(value: float) -> str:
        return f"{value:.6f}"

    lines = [
        "# Flow-variant representation collision audit",
        "",
        "## Scope and integrity",
        "",
        "This audit used only exact saved V4 `h(z, xi)` inputs. It trained no model, generated no state or rollout, and changed neither the 214-D schema nor oracle labels. Distances are normalized RMS distances using the exact outer-fold TRAIN-derived normalization; binary dimensions remain literal, matching the prior gate pipeline.",
        "",
        "The two collision neighbors were frozen from the prior local audit. The control was frozen before inspecting these cloud metrics as oracle-audit `boundary_pair_10`, an already `CLEARLY_SEPARATED` stable pair involving a non-collision zero state.",
        "",
        "## Results",
        "",
    ]
    for pair in pairs:
        metric = pair_metrics[pair["pair_id"]]
        static = next(row for row in static_rows if row["pair_id"] == pair["pair_id"])
        lines.extend(
            [
                f"### {pair['pair_id']}", "",
                f"- Fixed pair: `{pair['zero_state_id']}` (gate=0) vs `{pair['nonzero_state_id']}` (gate=1).",
                f"- Saved variants: {pair['variant_count_gate0']} / {pair['variant_count_gate1']}; exact matched seeds: {pair['matched_seed_count']}.",
                f"- Common state-static / union Flow-dependent dimensions: {static['common_state_static_dimensions']} / {static['union_flow_dependent_dimensions']}.",
                f"- Mean within-cloud distance: {fmt(metric['within_pooled_mean'])}; mean cross-cloud distance: {fmt(metric['cross_mean'])}; cross/within: {fmt(metric['cross_to_within_mean_ratio'])}.",
                f"- Opposite-state nearest-neighbor rate: {metric['opposite_state_nearest_neighbor_rate']:.2%}.",
                f"- Leave-one-out 1/3/5-NN accuracy: {metric['knn_1_accuracy']:.2%} / {metric['knn_3_accuracy']:.2%} / {metric['knn_5_accuracy']:.2%}.",
                f"- Energy distance {fmt(metric['energy_distance'])} (permutation p={metric['energy_permutation_p']:.4g}); MMD² {fmt(metric['mmd2_biased'])} (p={metric['mmd_permutation_p']:.4g}).",
                f"- Separating feature groups: {pair['cleanly_separating_feature_groups'] or 'none'}.",
                f"- Classification: **{pair['classification']}**.", "",
            ]
        )
    lines.extend(
        [
            "## Matched-seed interpretation", "",
            "The two primary collision pairs have no identical saved Flow seed IDs across their members, so an exact matched-seed comparison is unavailable and no index-based pseudo-pairing was used. Their valid comparison is distributional (all 64×64 cross distances versus within-cloud Flow variation). The control pair has 64 exact matched seeds; its detailed matched distances are saved separately.",
            "",
            "## Decision", "",
            f"Overall classification: **{overall}**.", "",
        ]
    )
    if overall == "REPRESENTATION_INFORMATION_PRESENT":
        next_step = "Replay all 64 saved variants of these two states through their existing source-group-held-out gate folds and report per-variant probability distributions; this directly tests whether the robust gate mistakes persist across xi despite the available input separation."
    elif overall == "TRUE_REPRESENTATION_COLLISION_SUPPORTED":
        next_step = "Audit one additional pre-existing stable collision candidate with saved variants to test whether this distributional collision repeats beyond these two states."
    else:
        next_step = "Repeat this exact saved-variant diagnostic on one additional pre-existing fixed pair matching the unresolved pair's pattern; do not collect new states yet."
    lines.extend(
        [
            "This conclusion is restricted to the two audited states and does not establish class-wide source-group generalization.",
            "",
            f"Smallest justified next experiment: {next_step}",
            "",
            "Permutation p-values are cloud-identity diagnostics, not independent-state evidence: the 214 coordinates are correlated and all 64 variants within a cloud share the same augmented state/static coordinates.",
        ]
    )
    (HERE / "collision_audit_report.md").write_text("\n".join(lines) + "\n")

    runtime = {
        "started_utc": START_UTC.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - START,
        "CPU_worker_limit": 4,
        "CPU_processes": 1,
        "CPU_threads_cap": 4,
        "GPU_used": False,
        "GPU_shards": 0,
        "models_trained": 0,
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "permutations_full_cloud_per_statistic": 1999,
        "permutations_feature_group_per_statistic": 499,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_CPU_count_visible": os.cpu_count(),
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    output_files = [
        "collision_audit_report.md", "audited_pairs.csv", "flow_variant_inventory.csv",
        "static_vs_flow_features.csv", "cloud_distance_statistics.csv", "nearest_neighbor_overlap.csv",
        "nonparametric_separability.csv", "matched_seed_distances.csv", "feature_group_overlap.csv",
        "control_pair_comparison.csv", "pca_projection.png", "sanity_checks.json", "runtime_statistics.json",
    ]
    inputs = [
        LOCAL / "per_state_failure_classification.csv", CV / "fold_manifest.json",
        DATA / "samples.npz", DATA / "feature_schema.json",
        ORACLE / "matched_pair_probability_differences.csv",
    ]
    manifest = {
        "experiment": "FLOW_VARIANT_REPRESENTATION_COLLISION_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall_classification": overall,
        "pair_classifications": {pair["pair_id"]: pair["classification"] for pair in pairs},
        "input_files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in inputs],
        "output_files": [{"path": name, "sha256": sha256(HERE / name)} for name in output_files],
        "analysis_script": {"path": "run_collision_audit.py", "sha256": sha256(HERE / "run_collision_audit.py")},
        "constraints": {"models_trained": 0, "new_states": 0, "new_oracle_rollouts": 0, "GPU_shards": 0},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({"overall": overall, "pairs": manifest["pair_classifications"], "wall_s": runtime["wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
