"""Sequential/resumable worker for one of at most two parent-scheduled shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.shards not in (1, 2) or not 0 <= args.shard < args.shards:
        raise ValueError("this audit permits only one or two sequential GPU workers")
    inventory = json.loads((HERE / "training_jobs.json").read_text())
    jobs = [job for job in inventory["jobs"] if int(job["job_index"]) % args.shards == args.shard]
    log = HERE / f"training_worker_shard{args.shard}.jsonl"
    with log.open("a") as handle:
        for job in jobs:
            command = [
                sys.executable, str(HERE / "run_logo_gate.py"), "--horizon", str(job["H"]),
                "--model", job["model"], "--fold-id", job["fold_id"], "--seed", str(job["seed"]),
                "--device", args.device, "--resume",
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            record = {**job, "returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
