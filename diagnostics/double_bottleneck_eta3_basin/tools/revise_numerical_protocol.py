#!/usr/bin/env python3
"""Create protocol v2 after the pre-outcome numerical projector rejection."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    invalid = STUDY / "invalid_numeric_attempt_v1"
    old_protocol_path = invalid / "PREREGISTRATION_v1.json"
    old_jobs_path = invalid / "stage_a_jobs_v1.json"
    protocol = json.loads(old_protocol_path.read_text())
    old_jobs = json.loads(old_jobs_path.read_text())
    protocol["schema"] = "double_bottleneck_eta3_basin_preregistration_v2"
    protocol["registered_at"] = datetime.now().astimezone().isoformat()
    protocol["supersedes"] = {
        "protocol": str(old_protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha(old_protocol_path),
        "reason": "The canonical Clarabel solve returned Solved with valid linear/KKT certificates but speed excess 5e-10 to 8e-10, above the frozen 1e-10 external certificate. The v1 partial sweep stopped and is excluded.",
        "eta_design_changed": False,
        "population_changed": False,
        "domain_or_samples_changed": False,
        "classification_changed": False,
    }
    protocol["numerical_projection_retry"] = {
        "canonical_primary": "The unchanged shared_control.HardSafetyFilter is always called first.",
        "trigger": "Only CBFSolverError from the canonical numerical solve; never a task outcome or ordinary intervention.",
        "retry_1": "Clarabel solve of the identical Euclidean objective, identical linear CBF rows, and identical four speed SOCs with tighter numerical tolerances.",
        "retry_2": "If retry_1 is still rejected, independent SLSQP solve of the identical linear inequalities and four exact quadratic speed balls.",
        "certificate": "The original feasibility_tol=1e-9, speed_tol=1e-10, and optimality_tol=2e-6 remain unchanged; no slack, clipping, relaxation, fallback action, or feasible-set change.",
        "precedent": "Same-problem numerical retry pattern used by the retained Toy success-basin implementation.",
        "diagnostic_validation": "Previously failing eta job completed collision-free with one certified identical-problem retry; eta=0 used zero retries and reproduced the frozen timeout.",
        "canonical_source_modified": False,
    }
    protocol["invalid_attempt"] = {
        "directory": str(invalid.relative_to(ROOT)),
        "scientific_rows_excluded": True,
        "individual_eta_outcomes_used_to_change_design": False,
        "note": "A few progress-line outcomes were visible while diagnosing the exception. No eta/domain/sample/population/refinement rule was changed; all v2 jobs restart under one uniform numerical protocol.",
    }
    protocol["frozen_hashes"]["evaluator"] = sha(STUDY / "tools/run_jobs.py")
    protocol["frozen_hashes"]["identical_projection_retry"] = sha(
        STUDY / "tools/exact_projection_retry.py"
    )
    new_protocol_path = STUDY / "PREREGISTRATION.json"
    write_new(new_protocol_path, protocol)
    jobs = dict(old_jobs)
    jobs["schema"] = "double_bottleneck_eta3_jobs_v2"
    jobs["preregistration_sha256"] = sha(new_protocol_path)
    write_new(STUDY / "jobs/stage_a.json", jobs)
    print(
        json.dumps(
            {
                "preregistration_sha256": sha(new_protocol_path),
                "evaluator_sha256": protocol["frozen_hashes"]["evaluator"],
                "retry_sha256": protocol["frozen_hashes"]["identical_projection_retry"],
                "jobs": len(jobs["jobs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
