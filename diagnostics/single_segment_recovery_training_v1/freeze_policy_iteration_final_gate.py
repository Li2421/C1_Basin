from pathlib import Path
import hashlib
import json


D = Path(__file__).resolve().parent
validation = json.loads((D / "policy_iteration_validation.json").read_text())
calibration = json.loads((D / "policy_iteration_calibration.json").read_text())
final_manifest = json.loads((D / "final_test_manifest.json").read_text())
selection = validation["selection"]
selected = selection["selected_metrics"]
cal_metrics = calibration["metrics"]
cal_audit = calibration["break_budget_audit"]
policy_hash = selection["selected_policy_hash"]
if cal_metrics["policy_hash"] != policy_hash:
    raise RuntimeError("validation/calibration policy mismatch")
if not (
    selected["segment_statistics"]["entered_count"] > 0
    and selected["segment_statistics"]["successful_finite_segment_then_safety_count"] > 0
    and cal_audit["point_estimate_within_budget"]
    and cal_metrics["hard_safety"]["intact"]
):
    raise RuntimeError("predeclared final gate did not pass")
if final_manifest["outcomes_observed"] is not False:
    raise RuntimeError("final test is no longer unopened")
payload = {
    "schema": "single_segment_final_test_gate_v1",
    "status": "FROZEN_APPROVED_FOR_FINAL_TEST",
    "selected_policy_hash": policy_hash,
    "validation_complete": True,
    "calibration_complete": True,
    "test_data_used": False,
    "final_test_unopened": True,
    "validation_q": selected["q_learned"],
    "validation_finite_segment_successes": selected["segment_statistics"]["successful_finite_segment_then_safety_count"],
    "calibration_q": cal_metrics["q_learned"],
    "calibration_break_rate": cal_metrics["break_rate"],
    "calibration_break_ci95": cal_metrics["break_rate_ci95"],
    "calibration_break_uncertainty_insufficient": cal_audit["insufficient_support"],
    "entry_almost_everywhere_degeneracy_observed": bool(
        selected["degeneracy"]["enter_almost_everywhere"]
        or cal_metrics["degeneracy"]["enter_almost_everywhere"]
    ),
    "gate_note": "Final testing is protocol-authorized by the frozen point-estimate gate; neither break control nor entry reliability is certified.",
}
payload["content_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
output = D / "policy_iteration_final_test_gate.json"
if output.exists():
    raise RuntimeError("refusing to overwrite frozen final gate")
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
