#!/usr/bin/env python3
"""Create an immutable manifest containing only unfinished rollout jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    completed = set()
    for path in sorted(args.manifest.parents[1].glob(args.raw_glob)):
        completed.update(json.loads(line)["job_id"] for line in path.read_text().splitlines() if line.strip())
    pending = [job for job in source["jobs"] if job["job_id"] not in completed]
    output = {**source, "resume_of": str(args.manifest), "completed_before_resume": len(completed), "jobs": pending}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed": len(completed), "pending": len(pending), "total": len(source["jobs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
