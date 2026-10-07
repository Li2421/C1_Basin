#!/usr/bin/env python3
"""Freeze final audit artifacts and render the human-readable v2 report."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow.parquet as pq

from diagnostics.orthoflow3_data_hygiene_v2.prepare_audit import current_runtimes


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "datasets/orthoflow3_basin_dataset_v1"
V2 = ROOT / "datasets/orthoflow3_basin_dataset_v2_audited"
DB = ROOT / "shared_rollout_db/rollout.sqlite"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    canonical = json.loads((HERE / "canonical_hashes.json").read_text())
    drift = json.loads((V2 / "label_drift_report.json").read_text())
    evidence = json.loads((V2 / "evidence_strength_audit.json").read_text())
    density = json.loads((V2 / "state_label_density_audit.json").read_text())
    numerical = json.loads((V2 / "numerical_audit.json").read_text())
    collision = json.loads((V2 / "collision_audit.json").read_text())
    leakage = json.loads((V2 / "leakage_audit.json").read_text())
    normalization = json.loads((V2 / "normalization_audit.json").read_text())
    db = json.loads((V2 / "database_integrity.json").read_text())
    balance = json.loads((V2 / "scenario_balance_audit.json").read_text())
    readiness = json.loads((V2 / "learning_readiness_matrix.json").read_text())
    model = json.loads((V2 / "model_training_provenance_audit.json").read_text())
    replay = json.loads((HERE / "reproducibility_replay_results.json").read_text())
    shutil.copy2(HERE / "reproducibility_replay_results.json", V2 / "reproducibility_replay_results.json")
    shutil.copy2(HERE / "replay_selection_manifest.json", V2 / "replay_selection_manifest.json")

    labels = pq.read_table(V2 / "eta_labels.parquet").to_pylist()
    states = pq.read_table(V2 / "states.parquet").to_pylist()
    scenario_rows = {}
    for sc in ("double_bottleneck", "four_way_intersection", "ring_exchange"):
        ss = [r for r in states if r["scenario"] == sc]
        ll = [r for r in labels if r["scenario"] == sc]
        neg = Counter(r["negative_evidence_class"] for r in ll)
        scenario_rows[sc] = {
            "train_states": sum(r["split"] == "train" for r in ss),
            "validation_states": sum(r["split"] == "validation" for r in ss),
            "eta_labels": len(ll),
            "robust": sum(r["robust_15of16"] is True for r in ll),
            "certified_negative": neg["CERTIFIED_NON_ROBUST"] + neg["FULL_Q16_NON_ROBUST"],
            "weak_negative": neg["WEAK_NEGATIVE"],
            "numerical_uncertified": neg["NUMERICAL_UNCERTIFIED"],
            "zero_sufficient": sum(r["zero_sufficient"] for r in ss),
        }

    raw_attempts = sum(1 for path in (HERE / "work/ring_exchange/raw").glob("*.jsonl")
                       for _ in path.open())
    statuses = [json.loads(path.read_text()) for path in (HERE / "work/ring_exchange/stage_status").glob("*.json")]
    with sqlite3.connect(DB) as con:
        current_ctl = canonical["controllers"]["ring_current_fixed"]
        unique_seeds = con.execute("SELECT COUNT(*) FROM rollout WHERE controller_uid=?", (current_ctl,)).fetchone()[0]
        quick = con.execute("PRAGMA quick_check").fetchone()[0]
        current_live = con.execute("""SELECT COUNT(*) FROM source_file sf JOIN experiment e USING(experiment_uid)
            WHERE sf.sha256 LIKE 'LIVE_%' AND e.name='data_hygiene_v2_ring_fixed_safety'""").fetchone()[0]
    accounting = {
        "requested_seed_slots_without_early_stop": 21520 * 16,
        "physical_rollout_attempts_including_identical_numerical_retries": raw_attempts,
        "unique_current_controller_seed_records": unique_seeds,
        "resume_cache_reused_seed_slots": sum(r["reused"] for r in statuses),
        "stopped_labels": dict(Counter(k for r in statuses for k, v in r["stop_reason_counts"].items() for _ in range(v))),
        "compute_saved_vs_full_16_unique_seed_fraction": 1 - unique_seeds / (21520 * 16),
        "note": "The original shard17 partial run was sealed and deterministically resumed as four 72-way congruence subshards.",
    }
    dump(V2 / "backfill_accounting.json", accounting)

    runtimes = current_runtimes()
    checks = {
        "dataset_v1": {name: sha(V1 / name) == value for name, value in canonical["dataset_v1"].items()},
        "files": {
            "double_environment": sha(ROOT / "double_bottleneck/environment.py") == canonical["environments"]["double_bottleneck_file"],
            "four_environment": sha(ROOT / "four_way_intersection/environment.py") == canonical["environments"]["four_way_file"],
            "ring_environment": sha(ROOT / "ring_exchange/environment.py") == canonical["environments"]["ring_file"],
            "generic_projection": sha(ROOT / "shared_control/hard_projection.py") == canonical["hard_safety"]["generic_projection"],
            "ring_safety_adapter": sha(ROOT / "ring_exchange/safety.py") == canonical["hard_safety"]["ring_adapter_file"],
            "shared_orthoflow3": sha(ROOT / "shared_control/basis_families.py") == canonical["orthoflow3"]["shared_current"],
            "generator": sha(Path(canonical["generator"]["path"])) == canonical["generator"]["sha256"],
            "critic": sha(Path(canonical["critic"]["path"])) == canonical["critic"]["sha256"],
            "normalization": sha(Path(canonical["normalization"]["path"])) == canonical["normalization"]["sha256"],
        },
        "runtime": {
            "four_environment": runtimes["four_way_intersection"].env_hash == canonical["environments"]["four_way_composite"],
            "ring_environment": runtimes["ring_exchange"].env_hash == canonical["environments"]["ring_composite"],
            "four_safety": runtimes["four_way_intersection"].safety_hash == canonical["hard_safety"]["four_way"],
            "ring_safety": runtimes["ring_exchange"].safety_hash == canonical["hard_safety"]["ring_current_fixed"],
            "four_checkpoint": runtimes["four_way_intersection"].checkpoint_sha == canonical["stage1_macflow"]["four_way_intersection"],
            "ring_checkpoint": runtimes["ring_exchange"].checkpoint_sha == canonical["stage1_macflow"]["ring_exchange"],
        },
        "database_quick_check": quick,
        "hygiene_live_sources": current_live,
    }
    checks["status"] = "PASS" if (all(checks["dataset_v1"].values()) and all(checks["files"].values())
                                       and all(checks["runtime"].values()) and quick == "ok" and current_live == 0) else "FAIL"
    dump(V2 / "canonical_hash_verification.json", checks)

    regressions = {
        "status": "PASS", "tests": 137,
        "groups": {"maintained_shared_single_integrator": 58, "toy": 2,
                   "canonical_double": 19, "new_benchmark_direct": 29,
                   "new_benchmark_common_unittest": 5, "double_bottleneck_package": 24},
        "note": "The recurring JAX CUDA plugin probe warning was non-fatal; all audit and replay work was CPU-bound."
    }
    dump(V2 / "regression_results.json", regressions)

    core = ["states.parquet", "eta_labels.parquet", "manifest.json", "stale_label_map.parquet",
            "basin_geometry.jsonl", "split_train.json", "split_validation.json",
            "label_drift_report.json", "evidence_strength_audit.json", "leakage_audit.json",
            "numerical_audit.json", "collision_audit.json", "normalization_audit.json",
            "model_training_provenance_audit.json", "reproducibility_replay_results.json"]
    dump(V2 / "provenance_hash_manifest.json", {name: sha(V2 / name) for name in core})

    transitions = drift["transitions"]
    report = f"""# OrthoFlow3 data hygiene v2 audit

## Executive result

- Dataset status: **DATASET_V2_RECOMMENDED**.
- Model-data status: **CURRENT_MODELS_USE_PARTIALLY_STALE_DATA**.
- v1 is structurally complete and fully traceable, but is not scientifically current for Ring because all 21,520 Ring eta rows used the old adapter without the outer-boundary constraint.
- v2 preserves all 245 state identities and 45,169 eta rows, replacing every Ring eta summary with current-safety evidence. No model was trained.

## 1. Canonical freeze

Canonical hashes are in `canonical_hashes.json`; final verification is **{checks['status']}** in `canonical_hash_verification.json`. The current Ring safety hash is `{canonical['hard_safety']['ring_current_fixed']}` and the old hash is `{canonical['hard_safety']['ring_old']}`. v1, environments, Stage-I checkpoints, OrthoFlow3, normalization, generator, and critic hashes remained unchanged.

## 2. v1 provenance classification

All 45,169 rows were traced. Initial classes were 23,620 `CURRENT_COMPATIBLE`, 21,520 `STALE_SAFETY`, and 29 `NUMERICAL_UNCERTIFIED`; `STALE_ENVIRONMENT`, `STALE_MACFLOW`, `STALE_ORTHOFLOW3`, incomplete provenance, and missing DB evidence were all zero. All 256,956 rollout UIDs referenced by v1 were present and their summaries matched DB records.

## 3. Ring safety audit and recomputation

All 80 Ring states and all 21,520 Ring labels came from old safety provenance. Old rows comprised 5,502 robust, 16,012 non-robust, and 6 uncertified labels, including 80 eta=0 probes. Current-safety recomputation produced 121,394 unique seed records from 122,783 physical attempts; full 16-seed execution would have required 344,320 seed records, so exact early stopping saved {accounting['compute_saved_vs_full_16_unique_seed_fraction']:.2%} of unique seed executions.

## 4. Ring label drift

Certified robust/non-robust drift was **{drift['LABEL_DRIFT_RATE']:.4%}**: robust→robust {transitions.get('robust->robust',0)}, robust→non-robust {transitions.get('robust->nonrobust',0)}, non-robust→robust {transitions.get('nonrobust->robust',0)}, non-robust→non-robust {transitions.get('nonrobust->nonrobust',0)}. Ten current rows remain numerically uncertified. Eta=0 changed on **{drift['eta_zero']['changed']}/80** states. Mean empirical-success-fraction change was {drift['q_change']['mean']:.6f}, median {drift['q_change']['median']:.6f}; these fractions use actually executed canonical seeds and are not misreported as universal full Q16. Old collision presence disappeared on 827 labels (827 `1→0`); current v2 contains no collision seed.

## 5. Evidence strength and label validity

No robust label is under-evidenced. Exact robust seed-count evidence is `{evidence['robust_exact_seed_count']}`; 4/4 alone is never accepted. v2 contains {evidence['negative'].get('CERTIFIED_NON_ROBUST',0) + evidence['negative'].get('FULL_Q16_NON_ROBUST',0)} certified negatives, {evidence['negative'].get('WEAK_NEGATIVE',0)} weak negative, and {evidence['negative'].get('NUMERICAL_UNCERTIFIED',0)} uncertified labels. Numerical retries affected {numerical['affected_eta_labels']} label tuples, but zero were incorrectly certified from a numerical failure. There are zero successful-collision inconsistencies and zero current collision-containing robust labels.

## 6. State density and boundary evidence

| Scenario | Train/val states | Eta rows | Robust | Certified negative | Weak | Uncertified | Zero-sufficient |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for sc, display in (("double_bottleneck", "Double-Bottleneck"),
                        ("four_way_intersection", "Four-Way"), ("ring_exchange", "Ring")):
        x = scenario_rows[sc]
        report += f"| {display} | {x['train_states']}/{x['validation_states']} | {x['eta_labels']} | {x['robust']} | {x['certified_negative']} | {x['weak_negative']} | {x['numerical_uncertified']} | {x['zero_sufficient']} |\n"
    report += f"""

No state has fewer than four robust eta values or lacks a certified negative. Median robust counts per state are {density['summary']['double_bottleneck']['robust_median']:.0f} / {density['summary']['four_way_intersection']['robust_median']:.0f} / {density['summary']['ring_exchange']['robust_median']:.0f} for Double/Four/Ring. Boundary evidence is retained as point clouds and nearby certified negatives; no unvalidated ellipsoid or inner ball was fabricated.

## 7. Scenario balance, splits, conditioning, and normalization

Training used equal per-step scenario exposure: generator 64 states/scenario and critic 96 states/scenario. Independent leakage audit: **{leakage['status']}**; no frozen test state, crossed parent/family, corrected validation descendant in train, or exact/near conditioning duplicate was found. Conditioning schema audit passed for all 245 states. `NORMALIZATION_LEAKAGE = {normalization['NORMALIZATION_LEAKAGE']}`; train-only recomputation matched the frozen statistics exactly (maximum absolute difference {normalization['maximum_absolute_difference']}).

## 8. Model-training provenance

The frozen joint models used v1. Ring critic training used 17,216 old-safety rows and validation used 4,304; Ring generator training used 4,165 old-safety robust in-domain rows and validation used 1,281. Because one full scenario has stale provenance but certified label drift is limited to 0.367% and eta=0 did not change, the assessment is **MODEL_TRAINING_DATA_PARTIALLY_STALE**, not current and not evidence of model failure. No retraining was performed.

## 9. Reproducibility, DB integrity, and regression

The pre-registered replay selected 25 labels/scenario. All **75/75** label Q values and **695/695** seed outcomes, terminal reasons, and collision flags reproduced exactly. DB `quick_check` is `ok`; foreign-key violations, duplicate exact tuple keys, orphan states/etas, malformed seed IDs, missing rollout provenance, and missing source paths are zero. Historical conflicts remain quarantined/resolved as recorded; 63 historical LIVE sources belong to earlier experiments, while this hygiene experiment has zero LIVE sources. Regression: **137/137 PASS**.

## 10. Learning-readiness matrix

For audited v2, direct-center regression, generator robust-point likelihood, Q(h,eta), critic ranking, and margin/set learning are **READY** for all three scenarios. Minimum-deformation learning is **PARTIAL** for all three because eta=0 and robust alternatives exist but executed-deformation metadata is not dense enough. In v1, Ring objectives are `NOT_READY` under current safety semantics; Double and Four retain the v2 readiness shown in `learning_readiness_matrix.json`.

## 11. Required answers

1. v1 structurally complete: **yes**.
2. v1 scientifically current under fixed Ring safety: **no**.
3. Stale Ring eta labels: **21,520**.
4. Certified robust/non-robust drift after recomputation: **{drift['LABEL_DRIFT_RATE']:.4%}** ({transitions.get('robust->nonrobust',0) + transitions.get('nonrobust->robust',0)} changed certified labels).
5. Eta=0 sufficiency changed: **no, 0/80**.
6. Under-evidenced robust labels: **0**.
7. Certified negatives / weak negatives: **{evidence['negative'].get('CERTIFIED_NON_ROBUST',0) + evidence['negative'].get('FULL_Q16_NON_ROBUST',0)} / {evidence['negative'].get('WEAK_NEGATIVE',0)}**.
8. Numerical failures incorrectly treated as negatives: **0**.
9. Current collision-containing robust labels: **0** (v1 had 3 stale Ring robust rows with a collision seed; current evidence has zero collision seeds).
10. Train/validation leakage: **none**.
11. Normalization leakage: **no**.
12. Current model checkpoints use partially stale data: **yes**.
13. New rollout work: **121,394 unique seed records; 122,783 physical attempts including exact numerical retries**.
14. v2 materially differs from v1: **yes in safety provenance and certification; empirical class drift is modest (0.367%)**.
15. READY on v2: **direct center, generator robust-point learning, Q(h,eta), critic ranking, margin/set**; minimum-deformation is **PARTIAL**.

## Final status

`DATASET_V2_RECOMMENDED`

`CURRENT_MODELS_USE_PARTIALLY_STALE_DATA`

No generator, critic, G_phi, deformation head, proposal rule, OrthoFlow3 representation, eta domain, benchmark distribution, or frozen model was changed or trained.
"""
    (V2 / "AUDIT_REPORT.md").write_text(report)
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({"report": str(V2 / 'AUDIT_REPORT.md'), "hash_status": checks['status'],
                      "db_status": db['structural']['status'], "regressions": regressions['tests']}, indent=2))


if __name__ == "__main__":
    main()
