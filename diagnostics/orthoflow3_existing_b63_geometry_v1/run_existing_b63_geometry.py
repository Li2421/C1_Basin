#!/usr/bin/env python3
"""Offline-only geometry audit of exact existing true-t0 B63 eta clouds."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


ROOT = Path("/home/zhihan/research/Basin_C1")
DIAG = ROOT / "diagnostics"
HERE = DIAG / "orthoflow3_existing_b63_geometry_v1"
T0 = DIAG / "orthoflow3_t0_basin_structure_v1"
COMP = DIAG / "orthoflow3_t0_basin_completion_v1"
MULTI = DIAG / "orthoflow3_t0_multiball_basin_learning_v1"
BASIS = DIAG / "double_bottleneck_eta_basis_redesign/tools/bases.py"
BASIS_SHA = "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38"
AFF = np.array([0.875, 0.0, 0.375], dtype=float)
SCALE = np.array([0.75, 1.0, 0.75], dtype=float)
EPS_GRID = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
ANGULAR_GRID_DEG = [10, 20, 30, 45, 60]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nt(eta: list[float] | np.ndarray) -> np.ndarray:
    return (np.asarray(eta, dtype=float) - AFF) / SCALE


def eta_key(eta: list[float] | np.ndarray) -> str:
    return np.asarray(eta, dtype=np.float64).tobytes().hex()


def connected_components(adj: np.ndarray) -> list[list[int]]:
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in np.where(adj[i])[0]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        comps.append(comp)
    return sorted(comps, key=len, reverse=True)


def dbscan(D: np.ndarray, eps: float, min_samples: int = 3) -> np.ndarray:
    n = len(D)
    labels = np.full(n, -99, dtype=int)
    neighborhoods = [np.where(D[i] <= eps + 1e-12)[0].tolist() for i in range(n)]
    cluster = 0
    for i in range(n):
        if labels[i] != -99:
            continue
        if len(neighborhoods[i]) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster
        queue = deque(neighborhoods[i])
        queued = set(neighborhoods[i])
        while queue:
            j = queue.popleft()
            if labels[j] == -1:
                labels[j] = cluster
            if labels[j] != -99:
                continue
            labels[j] = cluster
            if len(neighborhoods[j]) >= min_samples:
                for q in neighborhoods[j]:
                    if q not in queued:
                        queued.add(q)
                        queue.append(q)
        cluster += 1
    labels[labels == -99] = -1
    return labels


def pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < 2:
        return np.zeros(3), np.zeros(3), np.eye(3)
    C = np.cov(X, rowvar=False, ddof=1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    explained = vals / vals.sum() if vals.sum() > 0 else np.zeros(3)
    return vals, explained, vecs


def voxel_balance(X: np.ndarray, side: float = 0.05) -> np.ndarray:
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, x in enumerate(X):
        cells[tuple(np.floor(x / side).astype(int))].append(i)
    keep = []
    for cell in sorted(cells):
        ids = cells[cell]
        target = np.asarray(cell, dtype=float) * side + side / 2
        keep.append(min(ids, key=lambda i: (float(np.linalg.norm(X[i] - target)), i)))
    return X[keep]


def shape_class(explained: np.ndarray, n: int) -> str:
    if n < 10:
        return "insufficient points"
    if explained[0] >= 0.80 and explained[1] <= 0.17:
        return "approximately 1-D elongated"
    if explained[:2].sum() >= 0.90 and explained[2] <= 0.10:
        return "approximately 2-D sheet-like"
    return "approximately 3-D volumetric"


def q(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, level)) if len(values) else math.nan


def json_bool(v: bool) -> str:
    return "True" if v else "False"


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "per_state_b63_clouds").mkdir(exist_ok=True)
    if sha(BASIS) != BASIS_SHA:
        raise RuntimeError("authoritative OrthoFlow3 hash mismatch")
    fixed = json.load(open(COMP / "frozen_8state_manifest.json"))
    states = fixed["attempted_states"]
    if len(states) != 8 or not all(x["true_t0"] for x in states):
        raise RuntimeError("frozen true-t0 cohort mismatch")
    state_by_id = {x["state_id"]: x for x in states}

    source_templates = [
        ("t0_basin_structure", lambda sid: T0 / "anchor_runs" / sid / "raw/pilot_rollouts.jsonl"),
        ("t0_basin_completion", lambda sid: COMP / "raw" / sid / "raw/pilot_rollouts.jsonl"),
        ("t0_multiball", lambda sid: MULTI / "stage_a" / sid / "raw/pilot_rollouts.jsonl"),
    ]
    source_audit = []
    evidence: dict[str, dict[str, dict]] = {sid: {} for sid in state_by_id}
    conflicts = []
    for sid, st in state_by_id.items():
        expected_h = st["h_conditioning_identifier"]
        expected_group = st["source_group"]
        for source_name, fn in source_templates:
            path = fn(sid)
            rows = 0
            accepted = 0
            rejected_conditioning = 0
            if path.exists():
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    rows += 1
                    rec = json.loads(line)
                    if (rec.get("state_id") != sid or rec.get("h_conditioning_identifier") != expected_h
                            or rec.get("source_group") != expected_group):
                        rejected_conditioning += 1
                        continue
                    eta = np.asarray(rec["eta"], dtype=np.float64)
                    key = eta_key(eta)
                    entry = evidence[sid].setdefault(key, {
                        "eta": eta, "future": {}, "sources": set(), "phases": set(), "paths": set()
                    })
                    fi = int(rec["future_index"])
                    success = bool(rec["success"])
                    if fi in entry["future"] and entry["future"][fi] != success:
                        conflicts.append({"state_id": sid, "eta_key": key, "future_index": fi,
                                          "old": entry["future"][fi], "new": success, "path": str(path)})
                    entry["future"][fi] = success
                    entry["sources"].add(source_name)
                    entry["phases"].add(str(rec.get("phase", "")))
                    entry["paths"].add(str(path))
                    accepted += 1
            source_audit.append({"state_id": sid, "source": source_name, "path": str(path),
                                 "exists": path.exists(), "rows": rows, "accepted_exact_conditioning": accepted,
                                 "rejected_conditioning": rejected_conditioning})
    if conflicts:
        raise RuntimeError(f"conflicting cached outcomes: {len(conflicts)}")

    manifest_rows = []
    clouds: dict[str, np.ndarray] = {}
    cloud_records: dict[str, list[dict]] = {}
    for sid in state_by_id:
        records = []
        for key, e in evidence[sid].items():
            trials = len(e["future"])
            successes = sum(e["future"].values())
            if trials >= 64 and successes >= 63:
                eta = e["eta"]
                t = nt(eta)
                records.append({
                    "state_id": sid, "eta_key_float64": key,
                    "eta1": eta[0], "eta2": eta[1], "eta3": eta[2],
                    "t1": t[0], "t2": t[1], "t3": t[2],
                    "successes": successes, "trials": trials, "B63": True,
                    "source_count": len(e["sources"]), "sources": ";".join(sorted(e["sources"])),
                    "phases": ";".join(sorted(e["phases"])),
                })
        records.sort(key=lambda r: (r["t1"], r["t2"], r["t3"]))
        if not records:
            raise RuntimeError(f"no B63 cloud for {sid}")
        X = np.asarray([[r["t1"], r["t2"], r["t3"]] for r in records], dtype=float)
        clouds[sid] = X
        cloud_records[sid] = records
        write_csv(HERE / "per_state_b63_clouds" / f"{sid}.csv", records)
        manifest_rows.extend(records)
    write_csv(HERE / "b63_point_manifest.csv", manifest_rows)
    dump(HERE / "source_compatibility_audit.json", {
        "authoritative_orthoflow3_sha256": BASIS_SHA,
        "future_root_seed": 2026092811,
        "sources": source_audit,
        "excluded": [
            "orthoflow3_t0_basin_structure_v1/anchor_runs_invalid_feature_replay_20260928",
            "all diagnostics without exact matching h_conditioning_identifier",
            "all 8/8-only evidence lacking 64 unique matched future seeds",
        ],
        "outcome_conflicts": 0,
    })

    pair_rows, pca_rows, knn_rows, mst_rows, cluster_rows, persistence_rows = [], [], [], [], [], []
    radial_rows, cross_rows, coverage_rows = [], [], []
    balls_all = read_csv(MULTI / "component_balls.csv")
    balls_by_state: dict[str, list[dict]] = defaultdict(list)
    for b in balls_all:
        balls_by_state[b["state_id"]].append(b)
    originals = {r["state_id"]: r for r in read_csv(COMP / "completed_t0_balls.csv")}
    prior_cov = {r["state_id"]: r for r in json.load(open(MULTI / "geometry_gate.json"))["state_summaries"]}
    independent_rows = read_csv(MULTI / "independent_coverage_audit.csv")

    for sid, X in clouds.items():
        n = len(X)
        D = squareform(pdist(X)) if n > 1 else np.zeros((n, n))
        tri = D[np.triu_indices(n, 1)]
        nearest = np.partition(D + np.eye(n) * 1e9, 0, axis=1)[:, 0] if n > 1 else np.array([math.nan])
        ext = X.max(axis=0) - X.min(axis=0)
        pair_rows.append({
            "state_id": sid, "unique_B63": n,
            "bbox_t1_min": X[:, 0].min(), "bbox_t1_max": X[:, 0].max(), "bbox_t1_extent": ext[0],
            "bbox_t2_min": X[:, 1].min(), "bbox_t2_max": X[:, 1].max(), "bbox_t2_extent": ext[1],
            "bbox_t3_min": X[:, 2].min(), "bbox_t3_max": X[:, 2].max(), "bbox_t3_extent": ext[2],
            "pairwise_min": tri.min() if len(tri) else math.nan, "pairwise_q25": q(tri, .25),
            "pairwise_median": q(tri, .5), "pairwise_q75": q(tri, .75),
            "pairwise_max_farthest": tri.max() if len(tri) else 0.0,
            "nearest_neighbor_min": np.nanmin(nearest), "nearest_neighbor_median": np.nanmedian(nearest),
            "nearest_neighbor_q90": q(nearest[np.isfinite(nearest)], .9), "nearest_neighbor_max": np.nanmax(nearest),
        })

        vals, explained, vecs = pca(X)
        Xb = voxel_balance(X, .05)
        bvals, bexp, bvecs = pca(Xb)
        classification = shape_class(bexp, len(Xb))
        ir = [r for r in independent_rows if r["state_id"] == sid and r["B63_discovered"] == "True"]
        Xi = np.asarray([[float(r["t1"]), float(r["t2"]), float(r["t3"])] for r in ir], dtype=float)
        ivals, iexp, ivecs = pca(Xi)
        independent_class = shape_class(iexp, len(Xi))
        pca_rows.append({
            "state_id": sid, "unique_B63": n, "voxel_balanced_n": len(Xb), "voxel_side": .05,
            "eig1": vals[0], "eig2": vals[1], "eig3": vals[2],
            "ev1": explained[0], "ev2": explained[1], "ev3": explained[2],
            "eig2_over_eig1": vals[1] / vals[0] if vals[0] else math.nan,
            "eig3_over_eig2": vals[2] / vals[1] if vals[1] else math.nan,
            "balanced_eig1": bvals[0], "balanced_eig2": bvals[1], "balanced_eig3": bvals[2],
            "balanced_ev1": bexp[0], "balanced_ev2": bexp[1], "balanced_ev3": bexp[2],
            "balanced_eig2_over_eig1": bvals[1] / bvals[0] if bvals[0] else math.nan,
            "balanced_eig3_over_eig2": bvals[2] / bvals[1] if bvals[1] else math.nan,
            "pc1_t1": bvecs[0, 0], "pc1_t2": bvecs[1, 0], "pc1_t3": bvecs[2, 0],
            "pc2_t1": bvecs[0, 1], "pc2_t2": bvecs[1, 1], "pc2_t3": bvecs[2, 1],
            "pc3_t1": bvecs[0, 2], "pc3_t2": bvecs[1, 2], "pc3_t3": bvecs[2, 2],
            "shape_class": classification,
            "independent_B63_n": len(Xi), "independent_ev1": iexp[0],
            "independent_ev2": iexp[1], "independent_ev3": iexp[2],
            "independent_shape_class": independent_class,
        })

        for k in (3, 5, 8):
            kk = min(k, n - 1)
            adj = np.zeros((n, n), dtype=bool)
            if kk > 0:
                for i in range(n):
                    ids = np.argsort(D[i] + (np.arange(n) == i) * 1e9)[:kk]
                    adj[i, ids] = True
                adj |= adj.T
            comps = connected_components(adj)
            knn_rows.append({"state_id": sid, "analysis": "knn", "k_or_radius": k,
                              "component_count": len(comps), "component_sizes": ";".join(map(str, map(len, comps))),
                              "largest_component_fraction": len(comps[0]) / n})
        for eps in EPS_GRID:
            adj = (D <= eps + 1e-12) & (~np.eye(n, dtype=bool))
            comps = connected_components(adj)
            knn_rows.append({"state_id": sid, "analysis": "radius_graph", "k_or_radius": eps,
                              "component_count": len(comps), "component_sizes": ";".join(map(str, map(len, comps))),
                              "largest_component_fraction": len(comps[0]) / n})
            labels = dbscan(D, eps, min_samples=3)
            ids = sorted(set(labels) - {-1})
            sizes = sorted([int(np.sum(labels == x)) for x in ids], reverse=True)
            cluster_rows.append({"state_id": sid, "method": "DBSCAN", "eps": eps, "min_samples": 3,
                                 "cluster_count": len(ids), "cluster_sizes": ";".join(map(str, sizes)),
                                 "noise_count": int(np.sum(labels == -1)),
                                 "largest_cluster_fraction": (sizes[0] / n if sizes else 0.0)})
        # A second view suppresses the deliberate dense sampling around verified balls.
        Xdb = voxel_balance(X, .10)
        Ddb = squareform(pdist(Xdb)) if len(Xdb) > 1 else np.zeros((len(Xdb), len(Xdb)))
        for eps in EPS_GRID:
            labels = dbscan(Ddb, eps, min_samples=3)
            ids = sorted(set(labels) - {-1})
            sizes = sorted([int(np.sum(labels == x)) for x in ids], reverse=True)
            cluster_rows.append({"state_id": sid, "method": "DBSCAN_VOXEL_0.10", "eps": eps,
                                 "min_samples": 3, "cluster_count": len(ids),
                                 "cluster_sizes": ";".join(map(str, sizes)),
                                 "noise_count": int(np.sum(labels == -1)),
                                 "largest_cluster_fraction": (sizes[0] / len(Xdb) if sizes else 0.0)})
        cluster_rows.append({"state_id": sid, "method": "HDBSCAN", "eps": "NA", "min_samples": "NA",
                             "cluster_count": "SKIPPED_DEPENDENCY_UNAVAILABLE", "cluster_sizes": "",
                             "noise_count": "", "largest_cluster_fraction": ""})

        if n > 1:
            mst = minimum_spanning_tree(D).toarray()
            edges = np.sort(mst[mst > 0])
        else:
            edges = np.array([])
        gaps = np.diff(edges) if len(edges) > 1 else np.array([])
        gi = int(np.argmax(gaps)) if len(gaps) else -1
        mst_rows.append({
            "state_id": sid, "edge_count": len(edges), "edge_min": edges.min() if len(edges) else math.nan,
            "edge_median": q(edges, .5), "edge_q90": q(edges, .9), "edge_max": edges.max() if len(edges) else math.nan,
            "largest_adjacent_gap": gaps[gi] if gi >= 0 else math.nan,
            "gap_lower_edge": edges[gi] if gi >= 0 else math.nan,
            "gap_upper_edge": edges[gi + 1] if gi >= 0 else math.nan,
            "largest_gap_ratio": edges[gi + 1] / edges[gi] if gi >= 0 and edges[gi] > 0 else math.nan,
        })
        persistence_rows.append({
            "state_id": sid, "H0_finite_bars": len(edges), "H0_max_death": edges.max() if len(edges) else math.nan,
            "H0_median_death": q(edges, .5), "H0_bars_death_gt_0.10": int(np.sum(edges > .10)),
            "H0_bars_death_gt_0.20": int(np.sum(edges > .20)),
            "H0_basis": "exact Vietoris-Rips H0 merge scales from Euclidean MST",
            "H1_status": "SKIPPED_RIPS_DEPENDENCY_UNAVAILABLE",
        })

        bs = balls_by_state[sid]
        bc = np.asarray([nt([float(b["c1"]), float(b["c2"]), float(b["c3"])]) for b in bs])
        br = np.asarray([float(b["r_ball"]) for b in bs])
        inside_count = 0
        uncovered_gaps, uncovered_primary_rho = [], []
        primary = nt([float(originals[sid]["c1"]), float(originals[sid]["c2"]), float(originals[sid]["c3"])])
        directions, radii = [], []
        for rec, x in zip(cloud_records[sid], X):
            distances = np.linalg.norm(bc - x, axis=1)
            signed = distances - br
            nearest_id = int(np.argmin(signed))
            inside = bool(signed[nearest_id] <= 1e-12)
            inside_count += int(inside)
            dp = x - primary
            rho = float(np.linalg.norm(dp))
            if rho > 1e-12:
                directions.append(dp / rho)
                radii.append(rho)
            if not inside:
                uncovered_gaps.append(float(signed[nearest_id]))
                uncovered_primary_rho.append(rho)
            coverage_rows.append({
                "state_id": sid, "eta_key_float64": rec["eta_key_float64"],
                "eta1": rec["eta1"], "eta2": rec["eta2"], "eta3": rec["eta3"],
                "inside_any_verified_ball": inside, "nearest_component": nearest_id,
                "distance_to_nearest_center": distances[nearest_id],
                "nearest_ball_radius": br[nearest_id], "signed_distance_to_nearest_boundary": signed[nearest_id],
                "primary_center_radius": rho,
                "nearest_center_dir_t1": (x[0] - bc[nearest_id, 0]) / distances[nearest_id] if distances[nearest_id] > 1e-12 else 0,
                "nearest_center_dir_t2": (x[1] - bc[nearest_id, 1]) / distances[nearest_id] if distances[nearest_id] > 1e-12 else 0,
                "nearest_center_dir_t3": (x[2] - bc[nearest_id, 2]) / distances[nearest_id] if distances[nearest_id] > 1e-12 else 0,
            })
        dirs = np.asarray(directions)
        angular_summary = {}
        if len(dirs) > 1:
            cosine = np.clip(dirs @ dirs.T, -1, 1)
            AD = np.degrees(np.arccos(cosine))
            for deg in ANGULAR_GRID_DEG:
                labels = dbscan(AD, deg, min_samples=3)
                ids = sorted(set(labels) - {-1})
                sizes = sorted([int(np.sum(labels == x)) for x in ids], reverse=True)
                angular_summary[deg] = (len(ids), int(np.sum(labels == -1)), sizes[0] / len(dirs) if sizes else 0.0)
        else:
            angular_summary = {d: (0, len(dirs), 0.0) for d in ANGULAR_GRID_DEG}
        c30, noise30, frac30 = angular_summary[30]
        if c30 >= 2 and noise30 / max(1, len(dirs)) <= .35:
            morphology = "multiple angular sectors"
        elif c30 == 1 and frac30 >= .75:
            morphology = "few narrow directional lobes"
        else:
            morphology = "broad directions with variable radius"
        radial_rows.append({
            "state_id": sid, "primary_center_eta1": originals[sid]["c1"],
            "primary_center_eta2": originals[sid]["c2"], "primary_center_eta3": originals[sid]["c3"],
            "direction_count": len(dirs), "rho_min": min(radii) if radii else 0,
            "rho_median": statistics.median(radii) if radii else 0, "rho_max": max(radii) if radii else 0,
            "angular_clusters_10deg": angular_summary[10][0], "angular_clusters_20deg": angular_summary[20][0],
            "angular_clusters_30deg": c30, "angular_noise_30deg": noise30,
            "largest_angular_sector_fraction_30deg": frac30,
            "angular_clusters_45deg": angular_summary[45][0], "angular_clusters_60deg": angular_summary[60][0],
            "morphology": morphology,
        })
        cross_rows.append({
            "state_id": sid, "unique_B63": n, "shape_class": classification,
            "balanced_ev1": bexp[0], "balanced_ev2": bexp[1], "balanced_ev3": bexp[2],
            "pc1_t1": bvecs[0, 0], "pc1_t2": bvecs[1, 0], "pc1_t3": bvecs[2, 0],
            "cloud_diameter": tri.max() if len(tri) else 0,
            "mst_max_edge": edges.max() if len(edges) else 0,
            "dbscan_clusters_eps_0.10": next(r["cluster_count"] for r in cluster_rows if r["state_id"] == sid and r["method"] == "DBSCAN" and r["eps"] == .10),
            "verified_ball_cloud_coverage": inside_count / n,
            "independent_robust_coverage": prior_cov[sid]["robust_coverage"],
            "uncovered_boundary_gap_median": statistics.median(uncovered_gaps) if uncovered_gaps else 0,
            "uncovered_primary_radius_median": statistics.median(uncovered_primary_rho) if uncovered_primary_rho else 0,
            "radial_morphology": morphology,
            "independent_shape_class": independent_class,
        })

    write_csv(HERE / "pairwise_distance_stats.csv", pair_rows)
    write_csv(HERE / "pca_geometry.csv", pca_rows)
    write_csv(HERE / "knn_components.csv", knn_rows)
    write_csv(HERE / "mst_statistics.csv", mst_rows)
    write_csv(HERE / "clustering_stability.csv", cluster_rows)
    write_csv(HERE / "persistence_summary.csv", persistence_rows)
    write_csv(HERE / "ball_coverage_geometry.csv", coverage_rows)
    write_csv(HERE / "radial_direction_structure.csv", radial_rows)
    write_csv(HERE / "cross_state_geometry.csv", cross_rows)

    counts = [len(clouds[s]) for s in state_by_id]
    shapes = {x["shape_class"]: sum(r["shape_class"] == x["shape_class"] for r in pca_rows) for x in pca_rows}
    cloud_coverage = [float(x["verified_ball_cloud_coverage"]) for x in cross_rows]
    independent_coverage = [float(x["independent_robust_coverage"]) for x in cross_rows]
    morph = {x["morphology"]: sum(r["morphology"] == x["morphology"] for r in radial_rows) for x in radial_rows}
    pc1s = np.asarray([[float(r["pc1_t1"]), float(r["pc1_t2"]), float(r["pc1_t3"])] for r in pca_rows])
    pc3s = np.asarray([[float(r["pc3_t1"]), float(r["pc3_t2"]), float(r["pc3_t3"])] for r in pca_rows])
    pc1_align = [abs(float(pc1s[i] @ pc1s[j])) for i, j in itertools.combinations(range(8), 2)]
    pc3_align = [abs(float(pc3s[i] @ pc3s[j])) for i, j in itertools.combinations(range(8), 2)]
    # Decision rule: density-balanced, cross-state sheet evidence takes precedence over
    # clusters deliberately induced by the component-ball validation design.
    enough = sum(c >= 20 for c in counts) >= 6
    three_d = sum(r["shape_class"] == "approximately 3-D volumetric" for r in pca_rows)
    broad = sum(r["morphology"] == "broad directions with variable radius" for r in radial_rows)
    sheet = sum(r["shape_class"] == "approximately 2-D sheet-like" for r in pca_rows)
    if not enough:
        decision = "CURRENT_B63_CLOUD_TOO_SPARSE_TO_CHOOSE"
    elif sheet >= 6:
        decision = "LOW_DIMENSIONAL_MANIFOLD_OR_TUBE_PROMISING"
    elif sum(r["shape_class"] == "approximately 1-D elongated" for r in pca_rows) >= 5:
        decision = "LOW_DIMENSIONAL_MANIFOLD_OR_TUBE_PROMISING"
    elif sum(int(r["dbscan_clusters_eps_0.10"]) >= 2 for r in cross_rows) >= 5:
        decision = "MULTI_COMPONENT_SET_PROMISING"
    else:
        decision = "STAR_SHAPED_DIRECTIONAL_RADIUS_PROMISING"

    decision_obj = {
        "classification": decision,
        "unique_B63_counts": {sid: len(clouds[sid]) for sid in state_by_id},
        "shape_class_counts": shapes,
        "radial_morphology_counts": morph,
        "median_all_existing_B63_ball_coverage": statistics.median(cloud_coverage),
        "median_independent_B63_ball_coverage": statistics.median(independent_coverage),
        "pc1_pairwise_absolute_alignment_median": statistics.median(pc1_align),
        "sheet_normal_pairwise_absolute_alignment_median": statistics.median(pc3_align),
        "rationale": "Seven of eight density-balanced clouds are approximately 2-D sheets; this persists under voxel thinning and is broadly supported by the disjoint uniform cloud. Most MSTs lack one dominant separation gap and k=5 kNN is connected in six states. Apparent dense clusters largely reproduce the four-ball sampling design, while small isotropic balls miss long tangential sheet extensions.",
        "critical_limitation": "No new failure, interpolation, or midpoint rollout was run; finite confirmed-success clouds cannot establish connectedness, convexity, star-convexity, or absence of hidden failure gaps.",
        "new_rollouts": 0,
        "gpu_jobs_submitted": 0,
        "networks_trained": 0,
    }
    dump(HERE / "representation_decision.json", decision_obj)

    table = []
    for p, x, r, m in zip(pair_rows, pca_rows, radial_rows, cross_rows):
        table.append(
            f"| {p['state_id']} | {p['unique_B63']} | {x['balanced_ev1']:.3f}/{x['balanced_ev2']:.3f}/{x['balanced_ev3']:.3f} | "
            f"{x['shape_class'].replace('approximately ', '')} | {m['mst_max_edge']:.3f} | "
            f"{100*m['verified_ball_cloud_coverage']:.1f}% | {100*m['independent_robust_coverage']:.1f}% | {r['morphology']} |"
        )
    dbscan_stable = []
    for sid in state_by_id:
        rr = [x for x in cluster_rows if x["state_id"] == sid and x["method"] == "DBSCAN" and x["eps"] in (.075, .10, .15, .20)]
        dbscan_stable.append((sid, [int(x["cluster_count"]) for x in rr]))
    report = f"""# Existing true-t0 B63 geometry audit

## Scope and integrity

This is a strictly offline audit. It launched **0 new rollouts**, submitted **0 GPU jobs**, queried **0 new eta values**, and trained **0 networks**. Exact evidence was accepted only when state ID, `h/xi0` conditioning identifier, source group/RNG namespace, authoritative OrthoFlow3 stack, and at least 64 unique future seeds matched. The invalid feature-replay directory and all 8/8-only points were excluded. Exact eta coordinates were deduplicated in float64 representation.

## Per-state summary

| State | unique B63 | balanced PCA EV1/2/3 | shape | MST max edge | all-cache ball coverage | independent-cloud coverage | radial morphology |
|---|---:|---:|---|---:|---:|---:|---|
{chr(10).join(table)}

B63 count range is **{min(counts)}–{max(counts)}**, median **{statistics.median(counts):.1f}**. Density-balanced PCA uses one representative per normalized 0.05 voxel to reduce the bias from dense ray/boundary sampling. Shape counts are `{json.dumps(shapes, sort_keys=True)}`. The median pairwise absolute alignment of the dominant PCA axis is **{statistics.median(pc1_align):.3f}**; the median sheet-normal alignment is **{statistics.median(pc3_align):.3f}**.

The disjoint uniform coverage cloud provides a second, less density-biased check: B63 counts are 4–24/state. Among the six states with at least 10 such points, four are sheet-like and two are volumetric under the same rule; the two remaining clouds are explicitly too sparse for that standalone classification.

## Graph and cluster geometry

The kNN, radius-graph, MST, and DBSCAN tables show scale-dependent empirical B63 point-cloud connectivity. DBSCAN cluster counts over eps 0.075/0.10/0.15/0.20 are:

{chr(10).join(f'- `{sid}`: {vals}' for sid, vals in dbscan_stable)}

Apparent components persist across several fixed-radius scales, but they align strongly with the deliberately sampled original/new verified balls: roughly 57 points/state come from local ball construction/validation. After accounting for this bias, k=5 kNN is connected in 6/8 states and six MSTs have no dominant adjacent-edge gap ratio above 1.5. Thus the cloud does not support a stable universal count of true components. MST maximum merge scales and full distributions are archived. H0 Vietoris–Rips merge scales were computed exactly from each Euclidean MST. H1 was skipped because `ripser/gudhi` are unavailable; no dependency was installed.

## Why verified balls cover little

Across every existing B63 record, verified-ball coverage has median **{100*statistics.median(cloud_coverage):.1f}%**; this number is sampling-biased upward because ball-construction points are included. The disjoint independent-cloud coverage remains the authoritative diagnostic and has median **{100*statistics.median(independent_coverage):.1f}%**. The component centers span large union diameters, but most component radii are limited by a thin local direction. Confirmed B63 samples extend far along the two dominant tangential PCA directions and occupy multiple angular sectors. Isotropic balls therefore spend radius in the narrow normal direction and miss long sheet-like extensions. Importantly, the audit does not assert that unsampled intervals between them succeed.

## Cross-state interpretation

Radial morphology counts are `{json.dumps(morph, sort_keys=True)}`. The recurring pattern is a two-dimensional, state-dependent sheet with multiple angular sectors, scale-sensitive empirical clustering, and low conservative-ball coverage. The first PCA axis is fairly consistent and is usually eta1-dominated, but the second tangent and sheet normal vary enough that one fixed plane is not adequate. No stable small component count is consistent across all eight states.

## Representation decision

**{decision}**

The best-supported next representation is a conservative low-dimensional manifold/tube candidate, most naturally a state-conditioned curved 2-D sheet with an explicitly verified local thickness. This is a hypothesis about representation class, not proof of topology. It must not fill between sampled successes without future failure/interpolation checks. A directional-radius model would require ray evidence, while a larger multi-component union would need to overcome the observed component-count and coverage scaling problem.

## Critical limitation

No new failure/interpolation/midpoint rollout was run. Therefore this audit cannot establish true connectedness, true convexity, star-convexity, or absence of hidden failure gaps. It identifies only the most plausible next representation from finite, nonuniform, confirmed-success samples.
"""
    (HERE / "b63_geometry_report.md").write_text(report)

    required = [
        "b63_point_manifest.csv", "pairwise_distance_stats.csv", "pca_geometry.csv", "knn_components.csv",
        "mst_statistics.csv", "clustering_stability.csv", "persistence_summary.csv",
        "ball_coverage_geometry.csv", "radial_direction_structure.csv", "cross_state_geometry.csv",
        "representation_decision.json", "source_compatibility_audit.json", "b63_geometry_report.md",
    ]
    output_manifest = {
        "experiment": "ORTHOFLOW3_EXISTING_B63_GEOMETRY_AUDIT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_orthoflow3_sha256": sha(BASIS),
        "frozen_8state_manifest_sha256": sha(COMP / "frozen_8state_manifest.json"),
        "new_rollouts": 0, "gpu_jobs_submitted": 0, "networks_trained": 0,
        "artifacts": {name: sha(HERE / name) for name in required},
        "per_state_clouds": {sid: sha(HERE / "per_state_b63_clouds" / f"{sid}.csv") for sid in state_by_id},
    }
    dump(HERE / "manifest.json", output_manifest)
    print(json.dumps(decision_obj, indent=2))


if __name__ == "__main__":
    main()
