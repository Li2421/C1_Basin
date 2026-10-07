from pathlib import Path
import json

from decision_orchestration import analyze_full_loop_policy, calibration_break_budget_audit


D = Path(__file__).resolve().parent
manifest_path = D / "full_loop_manifests" / "calibration_policy_iteration" / "selected_eta.json"
manifest = json.loads(manifest_path.read_text())
result_dir = D / "runs" / "full_loop" / manifest["policy_hash"] / "calibration"
rows = [json.loads(path.read_text()) for path in sorted(result_dir.glob("single_segment_*.json"))]
metrics = analyze_full_loop_policy(rows, split="calibration", policy_hash=manifest["policy_hash"])
audit = calibration_break_budget_audit(metrics)
payload = {
    "schema": "single_segment_policy_iteration_calibration_v1",
    "status": "COMPLETE_FROZEN_NO_FURTHER_TUNING",
    "metrics": metrics,
    "break_budget_audit": audit,
    "cohort_note": "This cohort was previously inspected under a pre-iteration policy; no selection or tuning used this rerun.",
    "test_data_used": False,
}
(D / "policy_iteration_calibration.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
