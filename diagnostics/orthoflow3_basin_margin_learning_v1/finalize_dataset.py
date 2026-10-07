#!/usr/bin/env python3
"""Freeze the verified-ball dataset and the pre-registered budget stop.

This script performs no rollout or model training.  It combines the accepted
pilot/transfer labels with the labels built in this experiment and emits an
auditable underresolved handoff when the label-build continuation count exceeds
the protocol's 20,000-continuation stop threshold.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_basin_margin_learning_v1"
PREV = ROOT / "diagnostics/orthoflow3_large_margin_ball_transfer_v1"
PILOT = ROOT / "diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1"
LOWJ = ROOT / "diagnostics/orthoflow3_direct_eta_baseline_v1"
BASIS = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py"
EXPECTED_BASIS = "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38"
EXPECTED_LOWJ = "bd660db3ac501e5e77755af65cee5c01ba7d30810a4cf6001170bdbcebbb05d7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def b(v) -> bool:
    return str(v).lower() == "true"


def anchor_metadata() -> dict[str, dict]:
    out = {}
    for row in load(PILOT / "frozen_state_manifest.json")["states"]:
        out[row["state_id"]] = row
    for d in sorted((HERE / "anchor_builds").iterdir()):
        row = load(d / "frozen_state_manifest.json")["states"][0]
        out[row["state_id"]] = row
    return out


def label_row(*, state_id: str, anchor_id: str, kind: str, meta: dict,
              c1: float, c2: float, c3: float, radius: float,
              center_q64: float, provenance: str, mandatory_passed: int,
              mandatory_tested: int, false_inclusions: int) -> dict:
    return {
        "state_id": state_id,
        "split": meta["split"],
        "source_group": meta["source_group"],
        "source_trajectory": meta["source_trajectory"],
        "anchor_family": anchor_id,
        "label_kind": kind,
        "absolute_step": int(meta["absolute_step"]),
        "true_t0": int(meta["absolute_step"]) == 0,
        "temporal_neighbor": kind == "transferred",
        "offset_steps": meta.get("offset_steps", 0),
        "h_conditioning_identifier": meta["h_conditioning_identifier"],
        "flow_seed": meta["flow_seed"],
        "state_file": meta["state_file"],
        "state_sha256": meta["state_sha256"],
        "feature_source": meta.get("feature_source", "neighbor_features.npz"),
        "feature_index": meta.get("feature_index", ""),
        "feature_sha256": meta.get("feature_sha256", ""),
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "r_ball": radius,
        "center_Q64": center_q64,
        "domain_contained": True,
        "mandatory_inside_B63": mandatory_passed,
        "mandatory_inside_tested": mandatory_tested,
        "confirmed_internal_false_inclusions": false_inclusions,
        "provenance": provenance,
        "training_eligible": True,
    }


def main() -> None:
    assert sha(BASIS) == EXPECTED_BASIS
    selected = load(LOWJ / "selected_checkpoint.json")
    lowj_ckpt = Path(selected["checkpoint"])
    assert sha(lowj_ckpt) == EXPECTED_LOWJ

    meta_anchor = anchor_metadata()
    labels: list[dict] = []
    rejected: list[dict] = []

    # Accepted historical anchor labels.
    prev_anchor = load(PREV / "anchor_ball_manifest.json")["anchors"]
    for a in prev_anchor:
        if a["eligibility"] != "MARGIN_ELIGIBLE_ANCHOR":
            rejected.append({
                "state_id": a["state_id"], "split": a["split"],
                "source_group": a["source_group"], "r_ball": a["r_anchor"],
                "reason": "MARGIN_INELIGIBLE_R_LT_0.20", "stage": "prior_pilot",
            })
            continue
        m = meta_anchor[a["state_id"]]
        labels.append(label_row(
            state_id=a["state_id"], anchor_id=a["state_id"], kind="anchor", meta=m,
            c1=float(a["c1"]), c2=float(a["c2"]), c3=float(a["c3"]),
            radius=float(a["r_anchor"]), center_q64=float(a["center_Q64"]),
            provenance=str(PILOT), mandatory_passed=int(a["independent_promoted_B63"]),
            mandatory_tested=int(a["independent_promoted_total"]), false_inclusions=0))

    # Accepted historical transferred labels.
    p_params = {x["neighbor_id"]: x for x in read_csv(PREV / "transferred_ball_parameters.csv")}
    p_valid = {x["neighbor_id"]: x for x in read_csv(PREV / "transferred_ball_validity.csv")}
    p_center = {x["neighbor_id"]: x for x in read_csv(PREV / "transfer_center_b63.csv")}
    p_meta = {x["neighbor_id"]: x for x in load(PREV / "neighbor_manifest.json")["neighbors"]}
    for sid, p in p_params.items():
        v, m, q = p_valid[sid], p_meta[sid], p_center[sid]
        assert v["status"] == "TRAINING_USABLE" and int(v["confirmed_internal_false_inclusions"]) == 0
        labels.append(label_row(
            state_id=sid, anchor_id=p["anchor_state_id"], kind="transferred", meta=m,
            c1=float(p["c1"]), c2=float(p["c2"]), c3=float(p["c3"]),
            radius=float(p["r_transfer"]), center_q64=float(q["Q64"]),
            provenance=str(PREV), mandatory_passed=int(v["mandatory_B63_passed"]),
            mandatory_tested=int(v["mandatory_tested"]),
            false_inclusions=int(v["confirmed_internal_false_inclusions"])))

    # New expensive anchors.  Three were valid balls but below the frozen margin threshold.
    eligible_new = []
    for d in sorted((HERE / "anchor_builds").iterdir()):
        state_id = d.name
        ball = read_csv(d / "conservative_ball_parameters.csv")[0]
        center = read_csv(d / "robust_centers.csv")[0]
        inside = load(d / "false_inclusion_summary.json")
        r = float(ball["r_ball"])
        m = meta_anchor[state_id]
        valid_geometry = (ball["status"] == "BALL_RESOLVED" and int(center["successes"]) >= 63
                          and int(inside["robust_false_inclusions"]) == 0)
        if not valid_geometry or r < 0.20:
            rejected.append({
                "state_id": state_id, "split": m["split"], "source_group": m["source_group"],
                "r_ball": r,
                "reason": "MARGIN_INELIGIBLE_R_LT_0.20" if valid_geometry else "INVALID_GEOMETRY",
                "stage": "this_experiment",
            })
            continue
        eligible_new.append(state_id)
        labels.append(label_row(
            state_id=state_id, anchor_id=state_id, kind="anchor", meta=m,
            c1=float(ball["c1"]), c2=float(ball["c2"]), c3=float(ball["c3"]), radius=r,
            center_q64=float(center["Q64"]), provenance=str(d),
            mandatory_passed=int(inside["promoted_B63"]),
            mandatory_tested=int(inside["promoted_to_64"]),
            false_inclusions=int(inside["robust_false_inclusions"])))

    assert sorted(eligible_new) == sorted(["N_r002_m120", "N_r049_s159", "N_r126_s461", "N_r132_s238"])

    # New full-radius transferred labels.
    nroot = HERE / "new_anchor_transfers"
    n_params = {x["neighbor_id"]: x for x in read_csv(nroot / "transferred_ball_parameters.csv")}
    n_valid = {x["neighbor_id"]: x for x in read_csv(nroot / "transferred_ball_validity.csv")}
    n_center = {x["neighbor_id"]: x for x in read_csv(nroot / "transfer_center_b63.csv")}
    n_meta = {x["neighbor_id"]: x for x in load(nroot / "neighbor_manifest.json")["neighbors"]}
    for sid, p in n_params.items():
        v, m, q = n_valid[sid], n_meta[sid], n_center[sid]
        assert v["status"] == "TRAINING_USABLE"
        assert float(p["accepted_shrink_factor"]) == 1.0
        assert int(v["confirmed_internal_false_inclusions"]) == 0
        labels.append(label_row(
            state_id=sid, anchor_id=p["anchor_state_id"], kind="transferred", meta=m,
            c1=float(p["c1"]), c2=float(p["c2"]), c3=float(p["c3"]),
            radius=float(p["r_transfer"]), center_q64=float(q["Q64"]),
            provenance=str(nroot), mandatory_passed=int(v["mandatory_B63_passed"]),
            mandatory_tested=int(v["mandatory_tested"]),
            false_inclusions=int(v["confirmed_internal_false_inclusions"])))

    labels.sort(key=lambda x: ({"train": 0, "val": 1, "test": 2}[x["split"]], x["anchor_family"], x["state_id"]))
    assert len({x["state_id"] for x in labels}) == len(labels)
    counts = Counter(x["split"] for x in labels)
    families = defaultdict(set)
    for x in labels:
        families[x["split"]].add(x["anchor_family"])
    all_families = [families[s] for s in ("train", "val", "test")]
    assert not (all_families[0] & all_families[1] or all_families[0] & all_families[2] or all_families[1] & all_families[2])
    assert counts["train"] >= 24 and counts["val"] >= 8 and counts["test"] >= 8

    fields = list(labels[0])
    write_csv(HERE / "verified_ball_dataset.csv", labels, fields)
    write_csv(HERE / "rejected_ball_labels.csv", rejected,
              ["state_id", "split", "source_group", "r_ball", "reason", "stage"])
    write_csv(HERE / "t0_semantics_audit.csv", [{k: x[k] for k in (
        "state_id", "split", "anchor_family", "absolute_step", "true_t0", "temporal_neighbor",
        "offset_steps", "source_trajectory", "h_conditioning_identifier")} for x in labels],
        ["state_id", "split", "anchor_family", "absolute_step", "true_t0", "temporal_neighbor",
         "offset_steps", "source_trajectory", "h_conditioning_identifier"])

    split_doc = {
        "split_unit": "anchor/source family",
        "counts": dict(counts),
        "independent_anchor_groups": {s: len(families[s]) for s in ("train", "val", "test")},
        "families": {s: sorted(families[s]) for s in ("train", "val", "test")},
        "pairwise_family_overlap": {
            "train_val": sorted(families["train"] & families["val"]),
            "train_test": sorted(families["train"] & families["test"]),
            "val_test": sorted(families["val"] & families["test"]),
        },
        "leakage": False,
    }
    dump(HERE / "source_family_split.json", split_doc)
    radii = [float(x["r_ball"]) for x in labels]
    t0 = Counter(x["split"] for x in labels if x["true_t0"])
    kinds = Counter(x["label_kind"] for x in labels)
    stats = {
        "labels": len(labels), "counts": dict(counts), "label_kind_counts": dict(kinds),
        "independent_anchor_groups": split_doc["independent_anchor_groups"],
        "radii": {"mean": statistics.fmean(radii), "median": statistics.median(radii),
                  "min": min(radii), "max": max(radii)},
        "true_t0_counts": {s: t0[s] for s in ("train", "val", "test")},
        "intermediate_counts": {s: counts[s] - t0[s] for s in ("train", "val", "test")},
        "rejected_labels": len(rejected),
        "confirmed_internal_false_inclusions": sum(int(x["confirmed_internal_false_inclusions"]) for x in labels),
    }
    dump(HERE / "ball_label_statistics.json", stats)
    dump(HERE / "ball_dataset_manifest.json", {
        "schema": "orthoflow3_verified_conservative_ball_dataset_v1",
        "geometry": "normalized Euclidean balls in E_bridge",
        "label_semantics": "empirically verified conservative inner success set",
        "minimum_radius": 0.20, "B63": ">=63/64", "rows": len(labels),
        "dataset_csv_sha256": sha(HERE / "verified_ball_dataset.csv"),
        "source_family_split_sha256": sha(HERE / "source_family_split.json"),
        "orthoflow3_sha256": EXPECTED_BASIS,
        "training_gate_met": True,
        "training_started": False,
        "stop_reason": "label construction exceeded the pre-registered 20,000-new-continuation stop threshold",
    })

    dump(HERE / "frozen_lowj_reference.json", {
        "checkpoint": str(lowj_ckpt), "checkpoint_sha256": sha(lowj_ckpt),
        "expected_sha256": EXPECTED_LOWJ, "identity_verified": True,
        "selected_seed": selected["seed"], "best_epoch": selected["best_epoch"],
        "retrained": False, "evaluated_in_this_stage": False,
    })

    # Exact execution totals from the seven full anchors and the six transfer shards.
    anchor_runtime = [load(d / "runtime_statistics.json") for d in sorted((HERE / "anchor_builds").iterdir())]
    transfer_runtime = [load(p) for p in sorted((nroot / "work").glob("*/run_summary.json"))]
    anchor_new = sum(int(x["new_continuations"]) for x in anchor_runtime)
    anchor_steps = sum(int(x["physical_steps"]) for x in anchor_runtime)
    transfer_new = sum(int(x["new_continuations"]) for x in transfer_runtime)
    transfer_steps = sum(int(x["physical_steps"]) for x in transfer_runtime)
    reused = sum(max(0, int(x.get("reused_exact_tuples_used", 0)) - int(x["new_continuations"])) for x in anchor_runtime)
    round1_ids = {"S_r065_p24", "N_r132_s238", "S_r006_p06", "S_r021_p21"}
    round1 = [load(HERE / "anchor_builds" / x / "runtime_statistics.json") for x in round1_ids]
    round2 = [x for d, x in zip(sorted((HERE / "anchor_builds").iterdir()), anchor_runtime) if d.name not in round1_ids]
    critical = max(x["wall_time_seconds"] for x in round1) + max(x["wall_time_seconds"] for x in round2) + max(x["wall_time_seconds"] for x in transfer_runtime)
    runtime = {
        "cache_reused_exact_continuations": reused,
        "new_continuations": anchor_new + transfer_new,
        "full_anchor_new_continuations": anchor_new,
        "transfer_new_continuations": transfer_new,
        "physical_steps": anchor_steps + transfer_steps,
        "label_build_critical_path_wall_seconds": critical,
        "label_build_sum_worker_wall_seconds": sum(x["wall_time_seconds"] for x in anchor_runtime) + sum(x["wall_time_seconds"] for x in transfer_runtime),
        "training_time_seconds": 0,
        "evaluation_rollout_time_seconds": 0,
        "max_gpu_shards": 6, "cpu_threads_per_shard": 2, "max_concurrent_cpu_threads": 12,
        "gpu_memory_peak_mib": "not captured", "ram_allocation_peak_gib": 48,
        "label_build_stop_threshold": 20000,
        "label_build_stop_triggered": anchor_new + transfer_new > 20000,
        "overall_25000_continuation_cap_respected": anchor_new + transfer_new <= 25000,
        "overall_8000000_step_cap_respected": anchor_steps + transfer_steps <= 8000000,
        "optional_transfer_shell_skipped": True,
    }
    dump(HERE / "runtime_statistics.json", runtime)
    assert runtime["label_build_stop_triggered"]

    # Pre-registered stop: make absence of training/evaluation explicit, never fabricate metrics.
    common = {
        "status": "NOT_TRAINED_DUE_LABEL_BUDGET_STOP", "input_dim": 214,
        "architecture": [214, 128, 128, 3], "activation": "SiLU",
        "seeds": [17, 23, 41], "g_center_loss": "normalized center MSE",
        "g_margin_loss": "max(0, ||eta-c||/(r+eps)-0.80)^2",
        "training_started": False,
    }
    dump(HERE / "common_model_config.json", common)
    for arm in ("g_center", "g_margin"):
        d = HERE / arm
        d.mkdir(exist_ok=True)
        write_csv(d / "training_history.csv", [], ["status", "seed", "epoch", "loss"])
        dump(d / "seed_results.json", {"status": "NOT_TRAINED_DUE_LABEL_BUDGET_STOP", "runs": []})
        dump(d / "selected_checkpoint.json", {"status": "NOT_TRAINED_DUE_LABEL_BUDGET_STOP", "checkpoint": None})
        (d / "checkpoint_sha256.txt").write_text("NOT_TRAINED_DUE_LABEL_BUDGET_STOP\n")

    empty_specs = {
        "val_closedloop.csv": ["status", "arm", "seed", "state_id"],
        "test_geometric_predictions.csv": ["status", "arm", "state_id", "rho"],
        "test_64seed_closedloop.csv": ["status", "controller", "state_id", "successes"],
        "error_to_margin.csv": ["status", "rho_bin", "B63_rate", "mean_Q64"],
        "perturbation_robustness.csv": ["status", "arm", "state_id", "magnitude"],
        "robustness_vs_deformation.csv": ["status", "controller", "state_id", "J_def"],
        "fresh_wide_results.csv": ["status", "controller", "episode_id", "outcome"],
        "fresh_wide_pairwise.csv": ["status", "episode_id", "center_outcome", "margin_outcome"],
        "fresh_h_support_distance.csv": ["status", "episode_id", "nearest_h_distance"],
    }
    for name, fs in empty_specs.items():
        write_csv(HERE / name, [], fs)
    dump(HERE / "fresh_wide_manifest.json", {"status": "NOT_RUN_DUE_LABEL_BUDGET_STOP", "episodes": 0})

    stop_reason = (f"Verified label construction used {runtime['new_continuations']} new continuations, "
                   "exceeding the pre-registered 20,000-continuation label-stage stop threshold; "
                   "G_CENTER, G_MARGIN, TEST, perturbation, and fresh-WIDE stages were not started.")
    dump(HERE / "point_target_decision.json", {
        "classification": "POINT_TARGET_COMPARISON_UNDERRESOLVED", "reason": stop_reason,
        "g_lowj_identity_verified": True, "g_center_trained": False,
    })
    dump(HERE / "basin_supervision_decision.json", {
        "classification": "BASIN_MARGIN_LEARNING_UNDERRESOLVED", "reason": stop_reason,
        "dataset_gate_met": True, "g_center_trained": False, "g_margin_trained": False,
    })

    report = f"""# OrthoFlow3 basin-margin learning v1

## Outcome

**BASIN_MARGIN_LEARNING_UNDERRESOLVED**

The verified-ball dataset was successfully completed, but neural training was not started. Label construction used **{runtime['new_continuations']:,} new continuations**, which exceeds the protocol's explicit 20,000-continuation label-stage stop threshold. The overall experiment caps were still respected ({runtime['physical_steps']:,} physical steps; fewer than 25,000 continuations).

## Frozen verified-ball dataset

| split | labels | independent source families | true t=0 | intermediate/temporal |
|---|---:|---:|---:|---:|
| TRAIN | {counts['train']} | {len(families['train'])} | {t0['train']} | {counts['train']-t0['train']} |
| VAL | {counts['val']} | {len(families['val'])} | {t0['val']} | {counts['val']-t0['val']} |
| TEST | {counts['test']} | {len(families['test'])} | {t0['test']} | {counts['test']-t0['test']} |

There is zero source-family leakage. The dataset contains {kinds['anchor']} expensive-anchor labels and {kinds['transferred']} verified transfer labels. Ball radii have mean {stats['radii']['mean']:.6f}, median {stats['radii']['median']:.6f}, min {stats['radii']['min']:.6f}, and max {stats['radii']['max']:.6f}.

Seven new expensive anchors were attempted in frozen outcome-blind order. Four passed the frozen `r >= 0.20` usability criterion; three geometrically valid balls were rejected as too small (`S_r065_p24`: 0.127500, `S_r006_p06`: 0.021250, `S_r021_p21`: 0.170000). The four eligible anchors produced 16/16 full-radius (`s=1.00`) transferred labels. All transfer centers were 64/64 and all 64 mandatory interior points were 64/64; confirmed internal false inclusions were zero.

No label in this dataset is a true episode-start state (`absolute_step == 0`). It is therefore a state-conditioned diagnostic dataset, not a t=0 training dataset. Any later fresh-WIDE evaluation would be a genuine distribution/generalization test.

## Frozen reference and decisions

The frozen G_LOWJ checkpoint hash was verified as `{EXPECTED_LOWJ}` and was not retrained.

- Decision A: **POINT_TARGET_COMPARISON_UNDERRESOLVED**.
- Decision B: **BASIN_MARGIN_LEARNING_UNDERRESOLVED**.

No evidence about G_CENTER versus G_MARGIN has been generated in this run. In particular, verified set-valued supervision has not yet been shown to outperform center MSE; normalized error-to-radius, perturbation tolerance, J_def tradeoffs, and fresh-WIDE outcomes remain unevaluated.

## Runtime and next step

Label generation used {runtime['cache_reused_exact_continuations']} exact cached continuations and {runtime['new_continuations']:,} new continuations ({runtime['physical_steps']:,} physical steps). Critical-path rollout wall time was {runtime['label_build_critical_path_wall_seconds']/60:.2f} minutes with at most 6 GPU shards, 12 concurrent CPU threads, and 48 GiB scheduled RAM. GPU peak memory was not captured. Optional outside-shell work was skipped because it is not required for label usability and the stop threshold had already fired.

The smallest justified next step is a separately approved training/evaluation continuation using this now-frozen 25/10/9 dataset: train only G_CENTER and G_MARGIN with the predeclared three seeds, then run the frozen VAL/TEST/fresh-WIDE comparison. No Q, J, selector, or gate experiment should be added.
"""
    (HERE / "basin_margin_learning_report.md").write_text(report)

    # Final manifest is intentionally generated last.
    manifest_files = []
    excluded = {"manifest.json"}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.name in excluded or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(HERE))
        manifest_files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha(path)})
    dump(HERE / "manifest.json", {
        "schema": "orthoflow3_basin_margin_learning_v1",
        "status": "BASIN_MARGIN_LEARNING_UNDERRESOLVED",
        "stage_completed": "verified_ball_dataset",
        "dataset_gate_met": True,
        "training_started": False,
        "orthoflow3_sha256": sha(BASIS),
        "g_lowj_sha256": sha(lowj_ckpt),
        "controller_modified": False, "safety_projection_modified": False,
        "files": manifest_files,
    })


if __name__ == "__main__":
    main()
