"""Resume only the three interrupted GPU shards, entirely on CPU.

This preserves every valid completed tuple and never touches frozen sources or
previous result files.  It exists because the GPU jobs were intentionally
stopped when the user requested that VRAM not be saturated.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCES = (
    HERE / "candidate_resume_reuse.jsonl",
    HERE / "raw/candidate_pre_resume_0/records.jsonl",
    HERE / "raw/candidate_pre_resume_1/records.jsonl",
    HERE / "raw/candidate_recovery_resume_1/records.jsonl",
)
OUTPUT = HERE / "candidate_resume_after_gpu_stop_reuse.jsonl"


def main() -> None:
    rows: list[dict] = []
    seen: dict[tuple[str, tuple[float, ...], int], dict] = {}
    source_counts: dict[str, int] = {}
    for path in SOURCES:
        count = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["state_id"], tuple(row["eta"]), int(row["seed"]))
            if key in seen:
                raise AssertionError(("duplicate tuple across resume caches", key, path))
            seen[key] = row
            rows.append(row)
            count += 1
        source_counts[str(path.relative_to(HERE))] = count

    OUTPUT.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT.relative_to(HERE)),
                "unique_completed_tuples": len(rows),
                "source_counts": source_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
