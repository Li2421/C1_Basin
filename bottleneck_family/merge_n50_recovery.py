"""Merge paired large-N same-direction recovery shards with complete demos.

The source train/dev trajectories are copied exactly once. Only successful
train-only recovery trajectories from the shards are added. No test or
opposing-traffic evidence is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from new_benchmark_common.dataset import DATASET_SCHEMA, JointTransitionDataset


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(source: Path, shards: list[Path], output: Path):
    source, output = Path(source), Path(output)
    shards = [Path(p) for p in shards]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    n = int(source_manifest["scenario_config"]["num_agents"])
    if n not in (20, 50):
        raise ValueError("recovery merge requires N=20 or N=50")
    if source_manifest["schema"] != DATASET_SCHEMA or not source_manifest["complete"]:
        raise ValueError("source dataset is not complete")
    records = list(source_manifest["files"])
    seen_ids = {row["rollout_id"] for row in records}
    if len(seen_ids) != len(records):
        raise ValueError("duplicate source rollout IDs")
    accepted = {"LR":{"trajectories":0,"transitions":0},
                "RL":{"trajectories":0,"transitions":0}}
    shard_info = []
    for shard in shards:
        path = shard / "manifest.json"
        manifest = json.loads(path.read_text())
        for field in ("environment_fingerprint", "agent_order", "observation_shape", "action_shape",
                      "scenario_config"):
            if manifest[field] != source_manifest[field]:
                raise ValueError(f"shard/source mismatch in {field}: {shard}")
        report = manifest["extra_report"]
        recovery_parent = Path(report["source_dataset"]).resolve()
        if recovery_parent != source.resolve():
            # Multiple competence-only merges may be chained. Follow their
            # recorded ancestry and require every original parent record to
            # be present unchanged in the current source dataset.
            ancestor = source.resolve()
            visited = set()
            while ancestor != recovery_parent:
                if ancestor in visited:
                    raise ValueError("cycle in recovery source ancestry")
                visited.add(ancestor)
                ancestor_manifest = json.loads((ancestor / "manifest.json").read_text())
                copied_parent = ancestor_manifest.get("extra_report", {}).get("source_dataset")
                if copied_parent is None:
                    raise ValueError("recovery shard parent absent from source ancestry")
                ancestor = Path(copied_parent).resolve()
            parent_manifest = json.loads((recovery_parent/"manifest.json").read_text())
            parent_rows = {row["rollout_id"]: row for row in parent_manifest["files"]}
            source_rows = {row["rollout_id"]: row for row in source_manifest["files"]}
            if len(parent_rows) != len(parent_manifest["files"]) or any(
                    source_rows.get(rollout_id) != row for rollout_id, row in parent_rows.items()):
                raise ValueError("copied parent dataset records changed before recovery merge")
        if report["accepted"]["LR"] != report["accepted"]["RL"]:
            raise ValueError("recovery shard is not mirrored by direction")
        source_label = report.get("source_label", "uniform_recovery")
        if source_label not in ("uniform_recovery", "late_postgate_recovery", "early_queue_recovery",
                                "terminal_multi_recovery", "gate_local_recovery"):
            raise ValueError("unsupported recovery source label")
        fresh = [row for row in manifest["files"] if row["source"] == source_label]
        if len(fresh) != sum(report["accepted"][d]["trajectories"] for d in ("LR","RL")):
            raise ValueError("recovery count conflicts with shard report")
        for row in fresh:
            if row["split"] != "train" or row["rollout_id"] in seen_ids:
                raise ValueError("recovery must be unique train-only data")
            with np.load(shard / row["file"], allow_pickle=False) as data:
                initial = json.loads(str(data["initial_state_json"].item()))
                audit = json.loads(str(data["recovery_audit_json"].item()))
                metadata = json.loads(str(data["metadata_json"].item()))
            if audit["source_split"] != "train" or not audit["expert_success"]:
                raise ValueError("recovery source is not a successful train rollout")
            direction = initial["direction"]
            if direction not in accepted or metadata["direction"] != direction:
                raise ValueError("invalid recovery direction")
            accepted[direction]["trajectories"] += 1
            accepted[direction]["transitions"] += int(row["length"])
            records.append(row)
            seen_ids.add(row["rollout_id"])
        shard_info.append({"path":str(path),"sha256":_sha(path),
                           "case_start":report.get("case_start"),
                           "cases_per_category":report.get("cases_per_category"),
                           "model_trace_dir":report.get("model_trace_dir"),
                           "accepted":report["accepted"],
                           "teacher_failures":report["teacher_failures"]})
    if accepted["LR"] != accepted["RL"]:
        raise ValueError("merged recoveries are directionally imbalanced")
    for row in source_manifest["files"]:
        src = source / row["file"]
        dst = output / row["file"]
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    for shard in shards:
        manifest = json.loads((shard/"manifest.json").read_text())
        for row in manifest["files"]:
            if row["source"] != manifest["extra_report"].get("source_label", "uniform_recovery"):
                continue
            src = shard / row["file"]
            dst = output / row["file"]
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
    final = dict(source_manifest)
    final["files"] = records
    final["counts"] = {"split":{split:sum(row["split"]==split for row in records)
                                 for split in ("train","dev","test")},
                       "source":{kind:sum(row["source"]==kind for row in records)
                                 for kind in sorted({row["source"] for row in records}
                                                    | set(source_manifest["counts"]["source"]))}}
    final["extra_report"] = {"schema":f"n{n}_dense_postgate_competence_dataset_v1",
                              "source_dataset":str(source),
                              "source_manifest_sha256":_sha(source_manifest_path),
                              "shards":shard_info,"accepted_recovery":accepted,
                              "failed_recovery_candidates":sum(len(s["teacher_failures"])
                                                               for s in shard_info),
                              "opposing_demonstrations":0,
                              "selection":"all successful train-only mirrored recoveries; no direction oversampling"}
    (output/"manifest.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    # The production dataset reader checks array shapes, manifest/NPZ digests
    # and provenance. Load both splits before any model uses the merge.
    for split in ("train","dev"):
        data = JointTransitionDataset(output,split)
        if data.environment_fingerprint != final["environment_fingerprint"]:
            raise ValueError("merged dataset fingerprint failed reader validation")
    summary = {"schema":f"n{n}_dense_postgate_merge_v1","output":str(output),
               "manifest_sha256":_sha(output/"manifest.json"),
               "counts":final["counts"],"accepted_recovery":accepted,"shards":shard_info}
    (output.parent/"merge_report.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--shard",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(merge(args.source,args.shard,args.output),indent=2),flush=True)


if __name__ == "__main__":
    main()
