"""Reuse the audited 214->64->64->1 cost-sensitive head trainer for S/L."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BASE = Path("/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1")
sys.path.insert(0, str(BASE))
from train_decision_head_grid import preflight  # noqa: E402
from decision_orchestration import run_training_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    dataset, summary = preflight(args.dataset_manifest, args.output_directory)
    if dataset.head_kind != "entry_g" or dataset.input_dimension != 214:
        raise RuntimeError("Stage-1 local trainer requires an entry_g/214-D dataset")
    if not args.execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    trained = run_training_grid(dataset, args.output_directory)
    print(json.dumps({
        **summary, "status": "TRAINING_COMPLETE_REQUIRES_FULL_LOOP_VALIDATION",
        "candidate_count": trained["candidate_count"],
        "checkpoint_manifest": str((args.output_directory / "checkpoint_manifest.json").resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
