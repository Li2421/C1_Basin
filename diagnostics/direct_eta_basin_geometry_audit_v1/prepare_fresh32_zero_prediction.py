"""Freeze matched robust streams for fresh32 zero and learned-eta evaluation."""

from __future__ import annotations

import hashlib, json, os
from pathlib import Path


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> None:
    output = HERE / "fresh32_zero_prediction_plan.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite frozen plan")
    source = json.loads((HERE / "fresh32_manifest.json").read_text())
    source_root = Path("/home/zhihan/research/Basin_C1/diagnostics/gphi_structured_eta_fresh_wide_v1")
    result_path = source_root / "per_episode_results.csv"
    selected = []
    for episode in source["selected_episodes"]:
        index = int(episode["episode_index"])
        raw_path = source_root / "runs/raw/structured_eta" / f"episode_{index:04d}.json"
        trajectory_path = source_root / "runs/trajectories/structured_eta" / f"episode_{index:04d}.npz"
        row = json.loads(raw_path.read_text())
        if row["source_id"] != episode["source_id"] or row["trajectory_sha256"] != sha(trajectory_path):
            raise RuntimeError((index, "source eta trajectory mismatch"))
        selected.append({**episode, "frozen_eta_hat": [float(x) for x in row["eta_hat"]],
                         "frozen_eta_normalized_raw": [float(x) for x in row["eta_normalized_raw"]],
                         "eta_hat_source": "original exact matched Flow realization in frozen fresh-WIDE evaluation",
                         "eta_hat_source_record": str(raw_path), "eta_hat_source_record_sha256": sha(raw_path),
                         "eta_hat_source_trajectory": str(trajectory_path), "eta_hat_source_trajectory_sha256": sha(trajectory_path)})
    plan = {
        "schema": "direct_eta_fresh32_zero_prediction_plan_v1", "status": "FROZEN_BEFORE_ROLLOUT",
        "fresh32_manifest": str(HERE / "fresh32_manifest.json"),
        "fresh32_manifest_sha256": sha(HERE / "fresh32_manifest.json"),
        "conditions": ["ZERO", "CURRENT_G_ETA"],
        "robust_seeds": list(range(96020001, 96020065)),
        "rng_namespace_rule": "500000 + original episode_index",
        "flow_semantics": "episode_key=fold_in(PRNGKey(robust_seed),rng_namespace); step_key=fold_in(episode_key,absolute_step)",
        "prediction_semantics": "reuse each episode's frozen eta_hat from its original exact step-0 h; hold that single eta point fixed across all 64 new matched futures",
        "eta_hat_source_results": str(result_path), "eta_hat_source_results_sha256": sha(result_path),
        "zero_semantics": "eta=(0,0,0) fixed for the full episode",
        "B63": "success>=63/64; all 64 are evaluated for both conditions",
        "episode_count": 32, "matched_streams_per_condition_state": 64,
        "maximum_new_continuations": 4096, "maximum_physical_steps": 3481600,
        "selected_episodes": selected,
    }
    plan["content_sha256"] = canonical(plan)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n"); os.replace(tmp, output)
    print(json.dumps({"plan_sha256": sha(output), "content_sha256": plan["content_sha256"],
                      "continuations": plan["maximum_new_continuations"]}, indent=2))


if __name__ == "__main__":
    main()
