#!/usr/bin/env python3
"""Build a leakage-controlled canonical state×eta aggregate dataset from rollout.sqlite."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

H = Path(__file__).parent
ROOT = H.parents[1]
DBPATH = ROOT / "shared_rollout_db" / "rollout.sqlite"
SOURCES = {
    "Toy": {
        "root": H.parent / "orthoflow3_shared_eta_codebook_v1",
        "controller": "ctl_df736b67f6410260d87812e0a76946af7152d147c9a0a25189d1908a92567b34",
        "scenario": "ToyGiveWay",
    },
    "DB": {
        "root": H.parent / "orthoflow3_db_shared_mode_transfer_v1",
        "controller": "ctl_0ce9b25aa22d23cd3d07a01c61fd72f3c76cec4be3c053a3d4b8fbb6184f543d",
        "scenario": "DoubleBottleneck_4A",
    },
}
ETA_CANON_CENTER = np.array([.875, 0., .375])
ETA_CANON_SCALE = np.array([.75, 1., .75])
ETA_GROUP_RADIUS = .05


def dump(name, obj):
    (H / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(name, data):
    path = H / name
    fields = list(data[0]) if data else ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(data)


def hfloat(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) / 2**64


def union_find_groups(z, radius):
    n = len(z); parent = np.arange(n); size = np.ones(n, int)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        a, b = find(a), find(b)
        if a == b: return
        if size[a] < size[b]: a, b = b, a
        parent[b] = a; size[a] += size[b]
    for a, b in cKDTree(z).query_pairs(radius, output_type="ndarray"):
        union(int(a), int(b))
    roots = [find(i) for i in range(n)]
    labels = {}; out = np.zeros(n, int)
    for i, root in enumerate(roots):
        if root not in labels: labels[root] = len(labels)
        out[i] = labels[root]
    return out


def farthest_subset(indices, eta_z, cap, tag):
    if len(indices) <= cap: return list(indices)
    indices = list(indices)
    first = min(indices, key=lambda i: hashlib.sha256((tag + str(i)).encode()).hexdigest())
    chosen = [first]
    dist = np.linalg.norm(eta_z[indices] - eta_z[first], axis=1)
    while len(chosen) < cap:
        best_pos = int(np.argmax(dist + np.asarray([hfloat(tag + "|" + str(i)) * 1e-12 for i in indices])))
        best = indices[best_pos]; chosen.append(best)
        dist = np.minimum(dist, np.linalg.norm(eta_z[indices] - eta_z[best], axis=1))
        dist[best_pos] = -1
    return chosen


def subset_near(items, target, tag):
    """Deterministic subset-sum nearest a target; uses availability counts only."""
    items = sorted(items, key=lambda x: hashlib.sha256((tag + "|" + str(x[0])).encode()).hexdigest())
    limit = int(max(target * 2 + 1, target + max([w for _, w in items] or [0]) + 1))
    dp = {0: ()}
    for item, weight in items:
        old = list(dp.items())
        for total, chosen in old:
            new = total + int(weight)
            if new <= limit and new not in dp: dp[new] = chosen + (item,)
    best = min(dp, key=lambda total: (abs(total - target), hashlib.sha256((tag + "|sum|" + str(total)).encode()).hexdigest()))
    return set(dp[best])


def main():
    H.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DBPATH); con.row_factory = sqlite3.Row
    all_pairs, state_records, excluded = [], [], Counter()
    feature_arrays = {}
    uid_to_state = {}

    for scenario, spec in SOURCES.items():
        z = np.load(spec["root"] / "state_features.npz")
        ids = z["state_ids"].astype(str); splits = z["splits"].astype(str); features = z["features"].astype(np.float32)
        feature_arrays[scenario] = features
        for feature_index, (alias, split) in enumerate(zip(ids, splits)):
            candidates = con.execute('''SELECT DISTINCT a.state_uid,s.source_group,s.identity_quality
                FROM state_alias a JOIN state s USING(state_uid)
                WHERE a.alias=? AND s.scenario_uid=(SELECT scenario_uid FROM scenario WHERE name=?)
                  AND s.identity_quality IN ('CONDITIONING_EXACT','SOURCE_GROUP_STABLE')''', (alias, spec["scenario"])).fetchall()
            exact = [r for r in candidates if r["identity_quality"] == "CONDITIONING_EXACT"]
            if not exact:
                excluded[scenario + "_feature_state_without_conditioning_exact"] += 1; continue
            # The feature-bearing CONDITIONING_EXACT row defines the canonical state.
            canonical = exact[0]; source_group = canonical["source_group"]
            compatible_uids = sorted({r["state_uid"] for r in candidates if r["source_group"] == source_group})
            if not compatible_uids:
                compatible_uids = [canonical["state_uid"]]
            state_records.append({"scenario": scenario, "state_id": alias, "state_uid": canonical["state_uid"],
                                  "source_group": source_group, "state_split": split, "feature_index": feature_index,
                                  "compatible_state_uids": compatible_uids})
            for uid in compatible_uids:
                uid_to_state[(scenario, uid)] = (canonical["state_uid"], alias, source_group, split, feature_index)

        relevant = [uid for (sc, uid) in uid_to_state if sc == scenario]
        qmarks = ",".join("?" * len(relevant))
        query = f'''SELECT r.state_uid,r.eta_uid,r.seed_key,r.success,r.deadlock,r.timeout,r.collision,
                           e.eta1,e.eta2,e.eta3
                    FROM rollout r JOIN eta e USING(eta_uid)
                    WHERE r.controller_uid=? AND r.state_uid IN ({qmarks})
                      AND r.conflict_quarantined=0 AND r.numerical_failure=0
                      AND r.compatibility_quality='EXACT_REUSE' '''
        seed_records = defaultdict(dict); pair_eta = {}
        for r in con.execute(query, [spec["controller"], *relevant]):
            canon_uid, alias, source_group, split, feature_index = uid_to_state[(scenario, r["state_uid"])]
            key = (canon_uid, r["eta_uid"]); sk = r["seed_key"]
            value = (int(r["success"]), int(r["deadlock"]), int(r["timeout"]), int(r["collision"]))
            if sk in seed_records[key] and seed_records[key][sk] != value:
                seed_records[key][sk] = None
            else:
                seed_records[key][sk] = value
            pair_eta[key] = (float(r["eta1"]), float(r["eta2"]), float(r["eta3"]), alias, source_group, split, feature_index)
        for (state_uid, eta_uid), seeds in seed_records.items():
            if any(v is None for v in seeds.values()):
                excluded[scenario + "_cross_uid_seed_conflict_pairs"] += 1; continue
            vals = list(seeds.values()); eta1, eta2, eta3, alias, source_group, split, feature_index = pair_eta[(state_uid, eta_uid)]
            all_pairs.append({"scenario": scenario, "state_uid": state_uid, "state_id": alias,
                              "source_group": source_group, "state_split": split, "feature_index": feature_index,
                              "eta_uid": eta_uid, "eta1": eta1, "eta2": eta2, "eta3": eta3,
                              "controller_uid": spec["controller"], "n_trials": len(vals),
                              "n_success": sum(v[0] for v in vals), "n_deadlock": sum(v[1] for v in vals),
                              "n_timeout": sum(v[2] for v in vals), "n_collision": sum(v[3] for v in vals),
                              "seed_identities_known": True, "evidence_class": "SEED_EXACT"})

    # Spatial grouping is outcome-blind and joint across scenarios.
    eta_uids = sorted({r["eta_uid"] for r in all_pairs})
    eta_coord = {}
    for r in all_pairs: eta_coord[r["eta_uid"]] = np.array([r["eta1"], r["eta2"], r["eta3"]])
    eta_matrix = np.asarray([eta_coord[u] for u in eta_uids])
    eta_z = (eta_matrix - ETA_CANON_CENTER) / ETA_CANON_SCALE
    groups = union_find_groups(eta_z, ETA_GROUP_RADIUS)
    group_members = defaultdict(list)
    for uid, g in zip(eta_uids, groups): group_members[int(g)].append(uid)
    # Availability-stratified assignment prevents a scenario from accidentally having
    # no eta holdout. It uses pair existence only, never success outcomes.
    uid_group = {uid: int(g) for uid, g in zip(eta_uids, groups)}
    group_avail = {sc: Counter() for sc in SOURCES}
    for r in all_pairs: group_avail[r["scenario"]][uid_group[r["eta_uid"]]] += 1
    group_split = {}
    db_items = list(group_avail["DB"].items())
    db_test = subset_near(db_items, round(.15 * len([r for r in all_pairs if r["scenario"] == "DB"])), "db_test")
    db_val = subset_near([(g, w) for g, w in db_items if g not in db_test], round(.15 * len([r for r in all_pairs if r["scenario"] == "DB"])), "db_val")
    for g in db_test: group_split[g] = "test"
    for g in db_val: group_split[g] = "val"
    for g, _ in db_items:
        if g not in group_split: group_split[g] = "train"
    toy_total = len([r for r in all_pairs if r["scenario"] == "Toy"])
    fixed_test = sum(group_avail["Toy"][g] for g, s in group_split.items() if s == "test")
    fixed_val = sum(group_avail["Toy"][g] for g, s in group_split.items() if s == "val")
    unassigned = [(g, w) for g, w in group_avail["Toy"].items() if g not in group_split]
    toy_test = subset_near(unassigned, max(0, round(.15 * toy_total) - fixed_test), "toy_test")
    remain = [(g, w) for g, w in unassigned if g not in toy_test]
    toy_val = subset_near(remain, max(0, round(.15 * toy_total) - fixed_val), "toy_val")
    for g in toy_test: group_split[g] = "test"
    for g in toy_val: group_split[g] = "val"
    for g, _ in unassigned:
        if g not in group_split: group_split[g] = "train"
    for g in group_members:
        group_split.setdefault(g, "train")
    eta_meta = {uid: (int(g), group_split[int(g)]) for uid, g in zip(eta_uids, groups)}
    for r in all_pairs:
        r["eta_group"], r["eta_split"] = eta_meta[r["eta_uid"]]
        s, e = r["state_split"], r["eta_split"]
        r["regime"] = ({("train", "train"): "A_seen_state_seen_eta",
                         ("test", "train"): "B_unseen_state_seen_eta",
                         ("train", "test"): "C_seen_state_unseen_eta",
                         ("test", "test"): "D_unseen_state_unseen_eta"}.get((s, e), "OTHER"))
        r["empirical_q"] = r["n_success"] / r["n_trials"]
        r["n_eff"] = min(r["n_trials"], 16)
        r["b15_evaluable"] = r["n_trials"] >= 16
        r["b15"] = r["n_trials"] >= 16 and r["n_success"] / r["n_trials"] >= 15 / 16
        r["sampled_train"] = False
        r["sampled_val"] = False
        r["sampled_eval"] = False

    # Deterministic coverage-aware training selection: cap each state, then exact eta prevalence, then scenario total.
    coords_by_uid = {u: z for u, z in zip(eta_uids, eta_z)}
    for scenario in SOURCES:
        indices = [i for i, r in enumerate(all_pairs) if r["scenario"] == scenario and r["regime"] == "A_seen_state_seen_eta"]
        by_state = defaultdict(list)
        for i in indices: by_state[all_pairs[i]["state_uid"]].append(i)
        stage = []
        for state_uid, ix in by_state.items():
            zloc = np.asarray([coords_by_uid[all_pairs[i]["eta_uid"]] for i in ix])
            local_idx = farthest_subset(list(range(len(ix))), zloc, 128, scenario + state_uid)
            stage.extend(ix[j] for j in local_idx)
        by_eta = defaultdict(list)
        for i in stage: by_eta[all_pairs[i]["eta_uid"]].append(i)
        stage2 = []
        for eta_uid, ix in by_eta.items():
            stage2.extend(sorted(ix, key=lambda i: hashlib.sha256((scenario + all_pairs[i]["state_uid"] + eta_uid).encode()).hexdigest())[:256])
        if len(stage2) > 10000:
            zloc = np.asarray([coords_by_uid[all_pairs[i]["eta_uid"]] for i in stage2])
            keep = farthest_subset(list(range(len(stage2))), zloc, 10000, scenario + "train_total")
            stage2 = [stage2[j] for j in keep]
        for i in stage2: all_pairs[i]["sampled_train"] = True

        # Validation is simultaneous held-out state and eta. Eval caps are deterministic, not outcome-based.
        val_ix = [i for i, r in enumerate(all_pairs) if r["scenario"] == scenario and r["state_split"] == "val" and r["eta_split"] == "val"]
        val_ix = sorted(val_ix, key=lambda i: hashlib.sha256((scenario + all_pairs[i]["state_uid"] + all_pairs[i]["eta_uid"]).encode()).hexdigest())[:2000]
        for i in val_ix: all_pairs[i]["sampled_val"] = True
        for regime in ("A_seen_state_seen_eta", "B_unseen_state_seen_eta", "C_seen_state_unseen_eta", "D_unseen_state_unseen_eta"):
            ev = [i for i, r in enumerate(all_pairs) if r["scenario"] == scenario and r["regime"] == regime]
            ev = sorted(ev, key=lambda i: hashlib.sha256(("eval|" + scenario + all_pairs[i]["state_uid"] + all_pairs[i]["eta_uid"]).encode()).hexdigest())[:3000]
            for i in ev: all_pairs[i]["sampled_eval"] = True

    # Joint train-only eta affine normalization using unique eta values.
    train_eta_uids = sorted({r["eta_uid"] for r in all_pairs if r["sampled_train"]})
    train_eta = np.asarray([eta_coord[u] for u in train_eta_uids])
    eta_min, eta_max = train_eta.min(0), train_eta.max(0)
    eta_center = (eta_min + eta_max) / 2; eta_scale = np.maximum((eta_max - eta_min) / 2, 1e-6)
    state_norm = {}
    for scenario in SOURCES:
        state_ix = sorted({r["feature_index"] for r in all_pairs if r["scenario"] == scenario and r["sampled_train"]})
        x = feature_arrays[scenario][state_ix]
        state_norm[scenario] = {"mean": x.mean(0).tolist(), "std": np.maximum(x.std(0), 1e-6).tolist(), "dimension": x.shape[1]}

    # Parquet stores canonical keys and raw h vectors for full reproducibility.
    table_rows = []
    for r in all_pairs:
        q = dict(r)
        q["h_raw"] = feature_arrays[r["scenario"]][r["feature_index"]].astype(float).tolist()
        table_rows.append(q)
    pq.write_table(pa.Table.from_pylist(table_rows), H / "pair_table.parquet", compression="zstd")
    write_csv("pair_table_index.csv", [{k: v for k, v in r.items() if k != "h_raw"} for r in table_rows])
    np.savez_compressed(H / "feature_store.npz", toy=feature_arrays["Toy"], db=feature_arrays["DB"])

    state_groups = defaultdict(lambda: defaultdict(set))
    for r in state_records: state_groups[r["scenario"]][r["state_split"]].add(r["source_group"])
    state_split = {sc: {sp: sorted(v) for sp, v in splits.items()} for sc, splits in state_groups.items()}
    dump("state_split.json", {"method": "reuse frozen source-group-isolated selector splits", "groups": state_split,
                              "states": state_records})
    nearest = {}
    for sp in ("val", "test"):
        a = np.asarray([eta_z[i] for i, u in enumerate(eta_uids) if eta_meta[u][1] == sp])
        b = np.asarray([eta_z[i] for i, u in enumerate(eta_uids) if eta_meta[u][1] == "train"])
        d = cKDTree(b).query(a)[0] if len(a) and len(b) else np.asarray([])
        nearest[sp] = {"n": len(d), "min": float(d.min()) if len(d) else None, "median": float(np.median(d)) if len(d) else None,
                       "mean": float(d.mean()) if len(d) else None, "max": float(d.max()) if len(d) else None,
                       "q10": float(np.quantile(d, .1)) if len(d) else None}
    dump("eta_split.json", {"method": f"connected components at normalized radius <= {ETA_GROUP_RADIUS}; deterministic availability-only subset balancing to 70/15/15 per scenario",
                            "canonical_group_center": ETA_CANON_CENTER.tolist(), "canonical_group_scale": ETA_CANON_SCALE.tolist(),
                            "unique_eta": len(eta_uids), "groups": len(group_members),
                            "group_counts": Counter(group_split.values()), "eta_counts": Counter(eta_meta[u][1] for u in eta_uids),
                            "nearest_holdout_to_train": nearest,
                            "assignments": [{"eta_uid": u, "group": eta_meta[u][0], "split": eta_meta[u][1],
                                             "eta": eta_coord[u].tolist()} for u in eta_uids]})
    counts = defaultdict(Counter)
    for r in all_pairs:
        counts[r["scenario"]]["available_pairs"] += 1
        counts[r["scenario"]]["train_pairs"] += int(r["sampled_train"])
        counts[r["scenario"]]["val_pairs"] += int(r["sampled_val"])
        counts[r["scenario"]][r["regime"]] += 1
        counts[r["scenario"]]["pairs_n16"] += int(r["n_trials"] >= 16)
    manifest = {"database": str(DBPATH), "database_sha256": hashlib.sha256(DBPATH.read_bytes()).hexdigest(),
                "controllers": {s: x["controller"] for s, x in SOURCES.items()}, "basis_sha256": "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38",
                "counts": {s: dict(x) for s, x in counts.items()}, "excluded": dict(excluded),
                "eta_normalization": {"center": eta_center.tolist(), "scale": eta_scale.tolist(), "fit": "joint unique TRAIN eta range"},
                "state_normalization": state_norm, "training_weight": "min(n_trials,16)", "new_rollouts": 0}
    dump("dataset_manifest.json", manifest)
    dump("working_state.json", {"status": "DATASET_READY", "completed": ["database_audit", "canonical_aggregation", "state_split", "eta_spatial_split", "sampling"],
                                "next_action": "train joint and baselines", "new_rollouts": 0})
    write_csv("experiment_ledger.csv", [{"stage": "dataset", "status": "COMPLETE", "new_rollouts": 0,
                                          "detail": json.dumps(manifest["counts"], sort_keys=True)}])
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
