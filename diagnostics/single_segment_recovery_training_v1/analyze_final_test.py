from pathlib import Path
import json

from decision_orchestration import analyze_full_loop_policy


D = Path(__file__).resolve().parent
manifest_path = D / "full_loop_manifests" / "final_test" / "policy_iteration_eta.json"
manifest = json.loads(manifest_path.read_text())
result_dir = D / "runs" / "full_loop" / manifest["policy_hash"] / "final_test"
rows = [json.loads(path.read_text()) for path in sorted(result_dir.glob("single_segment_*.json"))]
if len(rows) != 200:
    raise RuntimeError(f"expected 200 final rows, got {len(rows)}")
metrics = analyze_full_loop_policy(
    rows, split="final_test", policy_hash=manifest["policy_hash"], allow_final_test=True
)
payload = {
    "schema": "single_segment_final_test_analysis_v1",
    "status": "COMPLETE_FROZEN_NO_POST_HOC_TUNING",
    "manifest_path": str(manifest_path),
    "manifest_content_sha256": manifest["content_sha256"],
    "metrics": metrics,
    "source_episode_count": len(rows),
    "test_based_adjustment_forbidden": True,
}
(D / "final_test_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
