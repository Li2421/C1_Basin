from pathlib import Path
import json
from decision_orchestration import analyze_full_loop_policy, select_policy_by_full_loop_validation, calibration_break_budget_audit

D = Path(__file__).resolve().parent
metrics = []
for manifest_path in sorted((D / "full_loop_manifests/validation").glob("*.json")):
    manifest = json.loads(manifest_path.read_text())
    result_dir = D / "runs/full_loop" / manifest["policy_hash"] / "validation"
    rows = [json.loads(path.read_text()) for path in sorted(result_dir.glob("single_segment_*.json"))]
    metrics.append(analyze_full_loop_policy(rows, split="validation", policy_hash=manifest["policy_hash"]))
selection = select_policy_by_full_loop_validation(metrics)
cal_manifest = json.loads((D / "full_loop_manifests/calibration/selected_eta.json").read_text())
cal_dir = D / "runs/full_loop" / cal_manifest["policy_hash"] / "calibration"
cal_rows = [json.loads(path.read_text()) for path in sorted(cal_dir.glob("single_segment_*.json"))]
calibration = analyze_full_loop_policy(cal_rows, split="calibration", policy_hash=cal_manifest["policy_hash"])
calibration_audit = calibration_break_budget_audit(calibration)
payload = {"candidates": metrics, "selection": selection, "calibration": calibration, "calibration_audit": calibration_audit}
(D / "validation_and_calibration_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"selection": selection, "calibration": calibration, "calibration_audit": calibration_audit}, indent=2, sort_keys=True))
