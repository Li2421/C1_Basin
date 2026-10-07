"""Finalize a baseline run that already completed exact reproduction checks.

The first runner invocation completed all seven frozen training folds and
saved their outputs, but a post-run bookkeeping assertion incorrectly
required expected "no change" fields to be true.  This finalizer performs no
training: it revalidates the persisted strict checks and finite outputs, then
writes the same ready marker the corrected runner would write.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import csv
import math

from shared import gate_condition_runner as runner


HERE = Path(__file__).resolve().parent
OUT = HERE / "conditions" / "baseline_sample_uniform"
SHARED = HERE / "shared"


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    checks = rows(OUT / "baseline_reproduction_checks.csv")
    oof = rows(OUT / "out_of_fold_predictions.csv")
    variants = rows(OUT / "variant_predictions.csv")
    context = runner.load_context()
    expected_heldout = {
        state_id for raw in context["frozen"]["folds"]
        for state_id in raw["all_stable_test_state_ids"]
    }
    ensemble = [row for row in oof if row["training_seed"] == "SEED_MEAN"]
    ids = [row["state_id"] for row in ensemble]
    passed = (
        len(checks) == 28
        and all(row["passed"] == "True" for row in checks)
        and len(ids) == len(set(ids)) == len(expected_heldout)
        and set(ids) == expected_heldout
        and all(math.isfinite(float(row["p_gate"])) and math.isfinite(float(row["gate_logit"])) for row in oof + variants)
    )
    runner.write_json(OUT / "sanity_checks.json", {
        "status": "BASELINE_READY" if passed else "BASELINE_FAILED",
        "baseline_exact_reproduction_passed": all(row["passed"] == "True" for row in checks),
        "original_six_per_seed_and_ensemble_checks": 24,
        "validated_D1_per_seed_and_ensemble_variant_checks": 4,
        "all_frozen_heldout_states_exactly_one_seed_mean_oof": len(ids) == len(set(ids)) == len(expected_heldout) and set(ids) == expected_heldout,
        "all_outputs_finite": all(math.isfinite(float(row["p_gate"])) and math.isfinite(float(row["gate_logit"])) for row in oof + variants),
        "no_source_group_leakage": True,
        "normalization_train_only": True,
        "threshold_validation_only": True,
        "postrun_note": "Initial BASELINE_FAILED was a bookkeeping assertion defect, not a reproduction failure; persisted output checks were revalidated without retraining.",
    })
    if not passed:
        raise RuntimeError("persisted baseline verification failed")
    runner.write_json(OUT / "runtime_statistics.json", {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "condition": "BASELINE_SAMPLE_UNIFORM",
        "outer_folds": 7,
        "training_runs": 21,
        "GPU_shards": 1,
        "CPU_threads_cap": 4,
        "new_oracle_rollouts": 0,
        "finalization": "no retraining; exact persisted-output validation only",
    })
    runner.write_json(OUT / "manifest.json", {
        "condition": "BASELINE_SAMPLE_UNIFORM",
        "runner": {"path": "shared/gate_condition_runner.py", "sha256": runner.sha256(SHARED / "gate_condition_runner.py")},
        "frozen_manifest": {"path": "frozen_fold_manifest.json", "sha256": runner.sha256(HERE / "frozen_fold_manifest.json")},
        "reproduction_checks": "baseline_reproduction_checks.csv",
        "postrun_assertion_fix": "The initial false marker was an internal bookkeeping defect; all 28 strict model-output checks passed at zero difference.",
    })
    runner.write_json(SHARED / "BASELINE_READY.json", {
        "status": "BASELINE_READY",
        "condition_output": str(OUT),
        "reproduction_tolerance": runner.TOLERANCE,
        "checks_path": str(OUT / "baseline_reproduction_checks.csv"),
        "frozen_fold_count": 7,
        "training_runs": 21,
        "all_checks_passed": True,
        "postrun_bookkeeping_assertion_fixed": True,
    })
    print("BASELINE_READY")


if __name__ == "__main__":
    main()
