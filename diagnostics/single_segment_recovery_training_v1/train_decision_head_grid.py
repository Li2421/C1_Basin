"""Preflight and explicitly launch a frozen decision-head training grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from decision_orchestration import build_head_dataset, file_hash, run_training_grid
from finalize_paired_branch_results import verify_semantic_hash


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError((path, "empty paired outcome file"))
    return rows


def preflight(dataset_manifest_path: Path, output_directory: Path) -> tuple[Any, dict[str, Any]]:
    frozen = json.loads(dataset_manifest_path.read_text())
    verify_semantic_hash(frozen, dataset_manifest_path)
    if frozen.get("status") != "COMPLETE_FROZEN_READY_FOR_TRAINING":
        raise RuntimeError("dataset is not complete/frozen for training")
    if frozen.get("all_results_complete") is not True or int(frozen.get("execution_error_count", -1)) != 0:
        raise RuntimeError("incomplete/errored branch rows cannot train a head")
    if frozen.get("final_test_used") is not False or frozen.get("training_launched") is not False:
        raise RuntimeError("dataset discipline/status mismatch")
    schema = str(frozen.get("schema"))
    if schema == "single_segment_finalized_head_dataset_v1":
        branch_manifest_path = Path(frozen["source_branch_manifest"])
        if file_hash(branch_manifest_path) != frozen["source_branch_manifest_sha256"]:
            raise RuntimeError("frozen branch manifest hash mismatch")
        branch_manifest = json.loads(branch_manifest_path.read_text())
        verify_semantic_hash(branch_manifest, branch_manifest_path)
        decision_rows = branch_manifest["decisions"]
        pass_ids = [str(frozen["pass_id"])]
    elif schema == "single_segment_merged_head_dataset_v1":
        decision_path = Path(frozen["merged_decision_rows_json"])
        if file_hash(decision_path) != frozen["merged_decision_rows_json_sha256"]:
            raise RuntimeError("merged decision-row hash mismatch")
        decision_payload = json.loads(decision_path.read_text())
        verify_semantic_hash(decision_payload, decision_path)
        if decision_payload.get("schema") != "single_segment_merged_decision_rows_v1":
            raise RuntimeError("invalid merged decision-row schema")
        decision_rows = decision_payload["decisions"]
        pass_ids = [str(value) for value in frozen["pass_ids"]]
        if decision_payload.get("pass_ids") != pass_ids:
            raise RuntimeError("merged decision/pass lineage mismatch")
        upstream = frozen.get("upstream_finalized_datasets", [])
        if len(upstream) != len(pass_ids):
            raise RuntimeError("merged upstream/pass count mismatch")
        for record in upstream:
            path = Path(record["path"])
            if file_hash(path) != record["sha256"]:
                raise RuntimeError("merged upstream finalized dataset changed")
            payload = json.loads(path.read_text())
            verify_semantic_hash(payload, path)
            if payload.get("content_sha256") != record["content_sha256"]:
                raise RuntimeError("merged upstream semantic lineage changed")
    else:
        raise RuntimeError(f"unsupported finalized dataset schema: {schema}")
    outcomes_path = Path(frozen["paired_branch_outcomes_jsonl"])
    if file_hash(outcomes_path) != frozen["paired_branch_outcomes_jsonl_sha256"]:
        raise RuntimeError("paired branch outcome hash mismatch")
    rows = load_jsonl(outcomes_path)
    if len(rows) != int(frozen["expected_result_count"]):
        raise RuntimeError("paired branch result count changed")
    dataset = build_head_dataset(
        decision_rows, rows,
        head_kind=str(frozen["head_kind"]),
        expected_futures_per_input=int(frozen["expected_futures_per_input"]),
    )
    if dataset.manifest != frozen["head_dataset_manifest"]:
        raise RuntimeError("rebuilt head dataset differs from frozen finalizer output")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError((output_directory, "refusing to train into a nonempty directory"))
    summary = {
        "status": "PREFLIGHT_PASS_TRAINING_NOT_LAUNCHED",
        "dataset_manifest_path": str(dataset_manifest_path.resolve()),
        "dataset_manifest_sha256": file_hash(dataset_manifest_path),
        "head_kind": dataset.head_kind,
        "pass_ids": pass_ids,
        "train_inputs": len(dataset.train_evidence),
        "validation_inputs": len(dataset.validation_evidence),
        "branch_rollouts": dataset.manifest["branch_rollouts"],
        "training_seeds": [17, 23, 41],
        "lambda_break_family": [0, 1, 3],
        "maximum_epochs": 1200,
        "validation_only_checkpoint_selection": True,
        "calibration_or_test_used": False,
    }
    return dataset, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true",
        help="explicitly launch the predeclared 3-seed x 3-lambda, 1200-epoch grid",
    )
    args = parser.parse_args()
    dataset, summary = preflight(args.dataset_manifest, args.output_directory)
    if not args.execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    result = run_training_grid(dataset, args.output_directory)
    print(json.dumps({
        **summary,
        "status": "TRAINING_COMPLETE_REQUIRES_FULL_LOOP_VALIDATION",
        "checkpoint_manifest": str((args.output_directory / "checkpoint_manifest.json").resolve()),
        "candidate_count": result["candidate_count"],
        "final_selected_policy": result["final_selected_policy"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
