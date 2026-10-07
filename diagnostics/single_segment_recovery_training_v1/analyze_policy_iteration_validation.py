from pathlib import Path
import json

from decision_orchestration import analyze_full_loop_policy, select_policy_by_full_loop_validation


D = Path(__file__).resolve().parent
manifest_dir = D / "full_loop_manifests" / "validation_policy_iteration"
metrics = []
for manifest_path in sorted(manifest_dir.glob("*.json")):
    manifest = json.loads(manifest_path.read_text())
    result_dir = D / "runs" / "full_loop" / manifest["policy_hash"] / "validation"
    rows = [json.loads(path.read_text()) for path in sorted(result_dir.glob("single_segment_*.json"))]
    metrics.append(
        analyze_full_loop_policy(rows, split="validation", policy_hash=manifest["policy_hash"])
    )
selection = select_policy_by_full_loop_validation(metrics)
payload = {
    "schema": "single_segment_policy_iteration_validation_v1",
    "status": "COMPLETE_FROZEN_SELECTION",
    "candidates": metrics,
    "selection": selection,
    "test_data_used": False,
}
output = D / "policy_iteration_validation.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(selection, indent=2, sort_keys=True))
