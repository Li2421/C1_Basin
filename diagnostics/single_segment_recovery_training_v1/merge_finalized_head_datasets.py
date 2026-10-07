"""Fail-closed merge of immutable paired-branch datasets across policy passes.

The merger performs no rollout and no training.  Every upstream finalized
dataset is reconstructed from its frozen branch manifest and typed outcome
JSONL before records are combined.  ``pass_id`` and downstream-policy hashes
remain part of each target lineage, so cost-to-go targets generated under
different policies are never silently treated as the same target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from decision_orchestration import build_head_dataset, content_hash, file_hash
from finalize_paired_branch_results import verify_semantic_hash


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError((path, "empty paired outcome file"))
    return rows


def _load_upstream(path: Path) -> tuple[dict, dict, list[dict], Any]:
    frozen = json.loads(path.read_text())
    verify_semantic_hash(frozen, path)
    if frozen.get("schema") != "single_segment_finalized_head_dataset_v1":
        raise RuntimeError((path, "upstream is not a single-pass finalized dataset"))
    if frozen.get("status") != "COMPLETE_FROZEN_READY_FOR_TRAINING":
        raise RuntimeError((path, "upstream is not complete/frozen"))
    if frozen.get("all_results_complete") is not True or int(frozen.get("execution_error_count", -1)) != 0:
        raise RuntimeError((path, "upstream contains incomplete/errored results"))
    if frozen.get("final_test_used") is not False:
        raise RuntimeError((path, "evaluation-only sources cannot enter head training"))

    branch_path = Path(frozen["source_branch_manifest"])
    if file_hash(branch_path) != frozen["source_branch_manifest_sha256"]:
        raise RuntimeError((path, "upstream branch manifest hash mismatch"))
    branch = json.loads(branch_path.read_text())
    verify_semantic_hash(branch, branch_path)
    if branch.get("content_sha256") != frozen["source_branch_manifest_content_sha256"]:
        raise RuntimeError((path, "upstream branch semantic hash mismatch"))
    if branch.get("pass_id") != frozen.get("pass_id") or branch.get("stage") != frozen.get("stage"):
        raise RuntimeError((path, "upstream pass/stage lineage mismatch"))

    outcomes_path = Path(frozen["paired_branch_outcomes_jsonl"])
    if file_hash(outcomes_path) != frozen["paired_branch_outcomes_jsonl_sha256"]:
        raise RuntimeError((path, "upstream paired outcomes hash mismatch"))
    rows = _load_jsonl(outcomes_path)
    if len(rows) != int(frozen["expected_result_count"]):
        raise RuntimeError((path, "upstream result count mismatch"))
    dataset = build_head_dataset(
        branch["decisions"], rows,
        head_kind=str(frozen["head_kind"]),
        expected_futures_per_input=int(frozen["expected_futures_per_input"]),
        require_validation=int(frozen["head_dataset_manifest"]["validation_decision_inputs"]) > 0,
    )
    if dataset.manifest != frozen["head_dataset_manifest"]:
        raise RuntimeError((path, "upstream dataset cannot be reproduced exactly"))
    return frozen, branch, rows, dataset


def _dataset_snapshot(dataset: Any, path: Path) -> None:
    _atomic_npz(
        path,
        train_features=dataset.train_features,
        validation_features=dataset.validation_features,
        train_r=np.asarray([item.r for item in dataset.train_evidence]),
        train_b=np.asarray([item.b for item in dataset.train_evidence]),
        train_pass_id=np.asarray([row["pass_id"] for row in dataset.train_rows]),
        train_downstream_policy_hash=np.asarray(
            [row["downstream_policy_hash"] for row in dataset.train_rows]
        ),
        validation_r=np.asarray([item.r for item in dataset.validation_evidence]),
        validation_b=np.asarray([item.b for item in dataset.validation_evidence]),
        validation_pass_id=np.asarray([row["pass_id"] for row in dataset.validation_rows]),
        validation_downstream_policy_hash=np.asarray(
            [row["downstream_policy_hash"] for row in dataset.validation_rows]
        ),
    )


def merge_finalized_datasets(
    upstream_manifest_paths: Sequence[Path], output_directory: Path
) -> dict[str, Any]:
    paths = [Path(path) for path in upstream_manifest_paths]
    if len(paths) < 2:
        raise ValueError("a multi-pass merge requires at least two finalized datasets")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("duplicate upstream finalized dataset")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError((output_directory, "refusing to overwrite merged dataset"))

    loaded = [_load_upstream(path) for path in paths]
    head_kinds = {frozen["head_kind"] for frozen, _, _, _ in loaded}
    future_counts = {int(frozen["expected_futures_per_input"]) for frozen, _, _, _ in loaded}
    pass_ids = [str(frozen["pass_id"]) for frozen, _, _, _ in loaded]
    if len(head_kinds) != 1 or len(future_counts) != 1:
        raise RuntimeError("cannot merge different head kinds/future-count protocols")
    if len(set(pass_ids)) != len(pass_ids):
        raise RuntimeError("each policy pass must have a distinct pass_id")
    source_split_hashes = {
        str(branch.get("source_split_manifest_sha256")) for _, branch, _, _ in loaded
    }
    if len(source_split_hashes) != 1 or "None" in source_split_hashes:
        raise RuntimeError("all policy passes must use the same frozen source split manifest")

    decisions = [dict(row) for _, branch, _, _ in loaded for row in branch["decisions"]]
    outcomes = [dict(row) for _, _, rows, _ in loaded for row in rows]
    head_kind = head_kinds.pop()
    future_count = future_counts.pop()
    dataset = build_head_dataset(
        decisions, outcomes, head_kind=head_kind,
        expected_futures_per_input=future_count,
    )

    # Verify the combined builder did not collapse pass/downstream lineages.
    expected_lineages: dict[str, int] = {}
    for decision in decisions:
        key = str(decision["downstream_policy_hash"])
        expected_lineages[key] = expected_lineages.get(key, 0) + 1
    if dataset.manifest["downstream_policy_lineage_counts"] != expected_lineages:
        raise AssertionError("combined downstream-policy lineage was altered")

    output_directory.mkdir(parents=True, exist_ok=True)
    decisions_path = output_directory / "merged_decision_rows.json"
    outcomes_path = output_directory / "paired_branch_outcomes.jsonl"
    snapshot_path = output_directory / "decision_dataset_snapshot.npz"
    decision_payload = {
        "schema": "single_segment_merged_decision_rows_v1",
        "pass_ids": pass_ids,
        "head_kind": head_kind,
        "decisions": decisions,
    }
    decision_payload["content_sha256"] = content_hash(decision_payload)
    _atomic_json(decisions_path, decision_payload)
    _atomic_jsonl(outcomes_path, outcomes)
    _dataset_snapshot(dataset, snapshot_path)

    upstream = [
        {
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "content_sha256": frozen["content_sha256"],
            "stage": frozen["stage"],
            "pass_id": frozen["pass_id"],
            "head_dataset_content_sha256": frozen["head_dataset_manifest"]["content_sha256"],
            "result_count": int(frozen["result_count"]),
        }
        for path, (frozen, _, _, _) in zip(paths, loaded, strict=True)
    ]
    result = {
        "schema": "single_segment_merged_head_dataset_v1",
        "status": "COMPLETE_FROZEN_READY_FOR_TRAINING",
        "head_kind": head_kind,
        "pass_ids": pass_ids,
        "upstream_finalized_datasets": upstream,
        "source_split_manifest_sha256": source_split_hashes.pop(),
        "merged_decision_rows_json": str(decisions_path.resolve()),
        "merged_decision_rows_json_sha256": file_hash(decisions_path),
        "paired_branch_outcomes_jsonl": str(outcomes_path.resolve()),
        "paired_branch_outcomes_jsonl_sha256": file_hash(outcomes_path),
        "result_count": len(outcomes),
        "expected_result_count": len(outcomes),
        "decision_count": len(decisions),
        "expected_futures_per_input": future_count,
        "dataset_snapshot": str(snapshot_path.resolve()),
        "dataset_snapshot_sha256": file_hash(snapshot_path),
        "head_dataset_manifest": dataset.manifest,
        "all_results_complete": True,
        "execution_error_count": 0,
        "train_validation_source_overlap": 0,
        "lineage_retained_not_merged": True,
        "final_test_used": False,
        "training_launched": False,
    }
    result["content_sha256"] = content_hash(result)
    _atomic_json(output_directory / "decision_dataset_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = merge_finalized_datasets(args.upstream_manifest, args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
