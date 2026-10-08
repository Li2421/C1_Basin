"""Give a recovery shard unique rollout IDs without changing its evidence.

Different frozen-prototype checkpoints may produce different physical states
at the same parent rollout and step. Their trajectory IDs must retain that
checkpoint distinction when the datasets are merged.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from new_benchmark_common.dataset import DatasetWriter, JointTransitionDataset
from .flow_dataset import GapFlowScenario
from .scenario import Config


def relabel(source: Path, output: Path, *, prefix: str):
    source, output = Path(source), Path(output)
    if not prefix or "/" in prefix:
        raise ValueError("prefix must be a nonempty filename-safe token")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["complete"] or manifest["counts"]["split"]["dev"] or manifest["counts"]["split"]["test"]:
        raise ValueError("relabel only a complete train-only recovery shard")
    config = Config(**manifest["scenario_config"])
    data = JointTransitionDataset(source, "train")
    writer = DatasetWriter(output, GapFlowScenario(config), scenario_config=manifest["scenario_config"])
    provenance = []
    for trajectory in data.trajectories:
        if trajectory.source != "late_postgate_recovery" or trajectory.recovery_audit is None:
            raise ValueError("only audited late post-gate recoveries can be relabeled")
        new_id = f"{prefix}_{trajectory.rollout_id}"
        metadata = dict(trajectory.metadata)
        metadata["relabelled_from_rollout_id"] = trajectory.rollout_id
        writer.add(replace(trajectory, rollout_id=new_id, metadata=metadata))
        provenance.append({"old":trajectory.rollout_id,"new":new_id})
    report = dict(manifest["extra_report"])
    report.update(relabelled_from=str(source),
                  relabelled_from_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  relabel_prefix=prefix,
                  relabel_map=provenance)
    writer.finalize(extra_report=report)
    verified = JointTransitionDataset(output, "train")
    if len(verified.trajectories) != len(data.trajectories):
        raise ValueError("relabelled dataset failed reader count verification")
    return {"source":str(source),"output":str(output),"trajectories":len(provenance),
            "manifest_sha256":hashlib.sha256((output/"manifest.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prefix",required=True)
    args = parser.parse_args()
    print(json.dumps(relabel(args.source,args.output,prefix=args.prefix),indent=2),flush=True)


if __name__ == "__main__":
    main()
