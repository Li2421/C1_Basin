"""Freeze the complete Stage-2 (H, model, LOGO fold, seed) job inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ready_path = HERE / "TRAINING_READY.json"
    folds_path = HERE / "fold_manifest.json"
    ready = json.loads(ready_path.read_text())
    if ready["fold_manifest_sha256"] != sha(folds_path):
        raise AssertionError("fold manifest changed after TRAINING_READY")
    folds = json.loads(folds_path.read_text())["folds"]
    jobs = []
    for fold in sorted(folds, key=lambda x: (int(x["H"]), x["fold_id"])):
        for model in ("LINEAR", "MLP_64x64"):
            for seed in (17,):
                jobs.append({"job_index": len(jobs), "H": int(fold["H"]), "model": model, "fold_id": fold["fold_id"], "seed": seed})
    obj = {
        "training_ready_sha256": sha(ready_path), "fold_manifest_sha256": sha(folds_path),
        "frozen_before_jobs_run": True, "job_count": len(jobs), "maximum_parallel_gpu_workers": 2,
        "jobs": jobs,
    }
    path = HERE / "training_jobs.json"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"job_count": len(jobs), "training_jobs_sha256": sha(path)}, indent=2))


if __name__ == "__main__":
    main()
